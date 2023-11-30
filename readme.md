# Detecting Anti-Israeli and Anti-Semitic Comments on Reddit

Reddit is the home for thousands of forums (`subreddits`) on various subjects that vary from mainstream to niche. It's slogan is "Dive into anything", and a as hinted, discussions on Reddit tend to be, relatively deeper than discussion on other social media networks. Another important feature of Reddit is the participants' anonimity. Each participant (`Redditor`) chooses their username at signup. This feature also lead to the more freedom when if comes to expressing extream and sometimes socialy unexeptable views.  
On multiple occations Reddit and it's CEO were critisized for the extant of hatered spread in some of the communities untouched or regulated, leading to a toxic echo chamber. One obvious victim of such dinamic is the jewish/Israeli community, that experienced a surge in anti-semitic comments following the October 7th events in Israel.  

In general, classification of comments on Reddit was challenging, as some of them aren't trivial. Table 1 present some titles and comments. Both in `/r/worldnews` and `/r/news` the title of the submission is the title of the article that is being refered to. It has no additional content. Therefore, comments were done by refering to the title or to the refered news article.  

| Title | Comment |
| ----- | ------- |
| Israeli forces arrest Al-Shifa Hospital director | What if this is true AND it’s also monstrous to bomb a hospital? |
| X's ad revenue to be donated to Israeli hospitals, Red Cross in Gaza: Musk | DESPERATION is thy name. |
| Israeli cabinet approves deal for return of 50 hostages in exchange for multi-day ceasefire | LETS FUCKIN GOOOO |
| Houthis say they'll continue attacks until 'demise of Israel' | I will file that under the FAFO category! 😬 |
| Houthis say they'll continue attacks until 'demise of Israel' | A perfect example of idiocracy. |
| Yemen's Houthis release footage of takeover of Israeli-linked cargo ship | GD rebel scum....flew right by the heli pad.....so rude |


In this project I build a classifier to detect anti-Israeli and anti-semitic comments. I then combine the model with Reddit's API, `PRAW`, in order to search through recent comments and downvote them for being anti-Israeli or anti-semitic.  

## Modeling
The modeling procedure is depicted in the following figure. As can be seen in the above table, looking at comments alone can be misleading. I therefore take the title and the comment and combine them together using the following rule `f'{title} ; {comment}`. BERT based models can handle sequences of up to a length of `512` tokens. In some cases the combinations of title and a comment surpass this window. In these cases I look only at the first part of the comment. A future work could extand the treatment for long sequences by pooling predictions for different parts, or by any other method.  
![Modeling](https://github.com/DavidHarar/Reddit/blob/main/plots/modeling.png)
For LLM I used the [Zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha). For classification I used [Roberta](https://huggingface.co/roberta-base). Label-spreading was done using the `[CLS]` token, using a `KNN` kernel. Voting ensemble was used between the LLM and the label spreading because neither of the method was good enough. After getting to the combined labeled data, I upscaled my data by asking the LLM to rephrase comments. I do it only once. The final version is of a fine-tuning using the labeled data, after I continue the pre-training, using the entire set of unlabeled data (it improves precision mostly on the right-most side of the predicted scores, in comparisson with the just-fine-tuned model). Fine-tuning was done by training the pre-trained model for 3 epochs on the labeled data.  

## Limitations and Further Work
I made accomadations on multiple occations to see the project through.
- Data: I used old Reddit's data using the Pushlift Data Dumps. Each subreddit includes two files, one for sumbissions and another for comments made for each sumbission, and for each comment. I didn't address comments to comments and basically trained a model using only comments made directly to the submission itself. 
- Model: As mentioned above, I didn't solve the issue of long strings. A further work would be able to solve it. 
- Labeled data: I manually labeled only 3k observations. In order to have a more robust data, one should label more examples. 


## Downvote and Legal
In Reddit there are two main social coins. One is `karma` and another is `awards`. The latter is basically diginal prizes given between redditors to express upreciation for meaningfull comments. Most of these awards cost money, and this project ignores them entirely. The other coin, `karma`, is basically the balance of upvotes and downvotes of the comments a redittor has made. Being downvoted for a hateful comment will inflict on the user's karma.  

Reddit API allows for downvote and upvote of comments and submissions directly form the API. Nevertheless, the usage of automated voting may be considered a violation of their policies, and hence **I do not recomend it, and any change to the provided code is on the modifier's responsibility**. For this reason, in this project I only reveal comments that are suspected of being antisemitic or anti-Israeli and take no action further than that.  
Regarding the use of Reddit'd Data in a machine learning model (see [3.2 Restrictions](https://www.redditinc.com/policies/data-api-terms?fbclid=IwAR3H0FWGTMVo1W0hJJhkBD31knspw36N5DvbO5RiBfz1bIl8lXk_tZLtKnE), Data API Terms), it is said that data that atchieved through Reddit's API won't be used to train ML models effective June 19,2023. The data used in this project was taken from [Reddit Pushlift](https://www.reddit.com/r/pushshift/), a community that aggregates Reddit's old data, from when it was perfectly legal to be used for training ML models. 


## How to get your Reddit API Key
In order to get such report, one has to have a Reddit API key. [This](https://www.reddit.com/wiki/api/) is a guide for how to get one. For convenience I summarize the steps below:
1. Read [Reddit’s Developer Terms](https://www.redditinc.com/policies/developer-terms) and [Data API Terms](https://www.redditinc.com/policies/data-api-terms). In section 4 it is specified that one should not use Reddit's data to train ML models. The restriciton is effective June 2023, while the data that was used to train the model is 10 years old. In the current project, no data is being collected nor being used for retraining the provided model.  
2. [Register](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164). Here you have to specify your interest. 
3. Get your personal API key from [here](https://old.reddit.com/prefs/apps/).