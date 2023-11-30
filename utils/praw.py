import re
from typing import List
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import praw
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

class AntiIsraelAPI():
  def __init__(self, reddit_credentials, model_information, config):
    # Init a Reddit session
    self.reddit = praw.Reddit(**reddit_credentials, check_for_async=False)
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    self.tokenizer = RobertaTokenizer.from_pretrained(model_information['tokenizer'])
    self.model = RobertaForSequenceClassification.from_pretrained(
        model_information['model'],
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False
        ).to(self.device)
    self.model_information = model_information
    self.config = config

  def vote(self, subreddit_name, limit, relevant_terms, posts_type):
    """
    run the entire process
    """
    # create an empty dictionary to store results
    results = {
        'sentences':[],
        'comments_ids':[],
        'title_id':[],
        'inputs':[],
        'scores':[],
        'prediction':[]
        }

    # get related posts
    posts = self._subreddit_get_posts(subreddit_name, limit, relevant_terms, posts_type)
    print(f'There are {len(posts)} posts about Israel.')

    # process posts
    for post in tqdm(posts, desc='Processing posts'):
      try:
        post_dataloader, comments_and_sentences = self._process_post(post, subreddit_name)
        prediction_results = self.predict(post_dataloader)
        post_results = {**comments_and_sentences,**prediction_results}

        # update results
        for k in results.keys():
          # print(f'{k}, {len(post_results[k])}, {type(post_results[k])}')
          results[k]+=post_results[k]
      except:
        continue

    return results


  def _subreddit_get_posts(self, subreddit_name, limit, relevant_terms, posts_type)->List:
    subreddit = self.reddit.subreddit(subreddit_name)
    if posts_type == 'new':
      posts = subreddit.new(limit=limit)
    if posts_type == 'hot':
      posts = subreddit.hot(limit=limit)

    # filter out irrelevant posts
    relevant_posts = []
    for post in posts:
      terms_included = 0
      title = str.lower(post.title).replace(f'/r/{subreddit_name} ','')
      for term in relevant_terms:
        if term in title and terms_included == 0:
          relevant_posts.append(post)
          terms_included+=1
        else:
          continue

    return relevant_posts

  def _process_post(self, post, subreddit_name)->DataLoader:
    # extract content from post
    post_and_comments = AntiIsraelAPI._post_get_comments_and_ids(post)

    # get title and comments
    post_title  = post_and_comments['title_text'][0].replace(f'/r/{subreddit_name} ','')
    post_id  = post_and_comments['title_id'][0]
    comments    = post_and_comments['comments_text']
    comments_ids= post_and_comments['comments_id']

    # duplicate `post_id`
    post_id = [post_id]*len(comments_ids)

    # process (clean comments and join titles and comments)
    comments = [AntiIsraelAPI.remove_hyperlinks(x) for x in comments]
    comments = [AntiIsraelAPI.trimmer(x) for x in comments]
    sentences = np.array([f'{post_title} ; {comment}' for comment in comments])

    # store sentences and comments IDs
    sentences_and_comments_ids = {'sentences':sentences.tolist(),'comments_ids':comments_ids, 'title_id':post_id}

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = self.tokenizer.encode_plus(
                            sent,                           # Sentence to encode.
                            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                            max_length = 512,               # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',          # Return pytorch tensors.
                      )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors and then to torch dataset.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size = 32
        )

    return dataloader, sentences_and_comments_ids

  def predict(self, dataloader):

    results = {
        'inputs':[],
        'scores':[]
        }

    for batch in dataloader:
      b_input_ids = batch[0].to(self.device)
      b_input_mask = batch[1].to(self.device)

      with torch.no_grad():
        scores = self.model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask)

        logits = scores.logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().tolist()


        results['inputs'] += b_input_ids.tolist()
        results['scores'] += probs

    results['prediction'] = np.where(np.array(results['scores'])>self.model_information['threshold'],1,0).tolist()

    return results


  @staticmethod
  def _post_get_comments_and_ids(post)->dict:
    """
    Extract content from a Reddit submission.

    inputs:
    post: Reddit Submission(id='182j32m')

    output:
    - dictionary with post ID, post title, comments ID and comments texts.

    Example:
    _post_get_comments_and_ids(post)

    """
    results = {
        'title_text':[],
        'title_id':[],
        'comments_text':[],
        'comments_id':[]
        }

    results['title_text'].append(post.title)
    results['title_id'].append(post.id)

    comments = [x for x in post.comments]
    results['comments_text'] = [comments[j].body for j in range(len(comments)-1)]
    results['comments_id'] = [comments[j].id for j in range(len(comments)-1)]

    return results

  @staticmethod
  def remove_hyperlinks(text):
    """Removes hyperlinks from text.

    Args:
      text: A string containing the text with hyperlinks.

    Returns:
      A string containing the text without hyperlinks.
    """
    # replace http to https
    text = text.replace('http://', 'https://')

    # Remove embedded hyperlinks.
    text = re.sub(r'\[(.*?)\]\(https:\/\/.*?\)', r'\1', text)

    # Remove pasted hyperlinks.
    text = re.sub(r'https?://\S+', '', text)

    return text

  @staticmethod
  def count_hyperlinks(text):
    """Counts the number of hyperlinks in text.

    Args:
      text: A string containing the text with hyperlinks.

    Returns:
      An integer representing the number of hyperlinks in the text.
    """
    # replace http to https
    text = text.replace('http://', 'https://')

    # Count embedded hyperlinks.
    hyperlinks = re.findall(r'https:\/\/.*?', text)

    # Return the total number of hyperlinks.
    return len(hyperlinks)

  @staticmethod
  def trimmer(text):
    """Removes extra characters in a row from text.
    """
    # remove trailing spaces
    lines = text.splitlines()

    # Remove trailing spaces from each line.
    clean_lines = [line.rstrip() for line in lines]

    # Join the lines back into a string.
    text = "\n".join(clean_lines)

    # Remove newlines in a row that have only one character in them.
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Remove extra spaces in a row.
    text = re.sub(r'\s+', ' ', text)

    # Remove multiple blank lines.
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Remove unnecessary whitespace.
    text = re.sub(r'\s+', ' ', text)

    # Remove extra dashes in a row.
    text = re.sub(r'--+', '--', text)

    # remove more than one \n in a row
    text = re.sub(r'\n+', '\n', text)

    # Compile the patterns into a regular expression.
    substrings = ['- ', '* ', ' -', ' *']
    for substring in substrings:
      text = text.replace(substring, '')

    return text
