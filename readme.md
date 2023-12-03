# Detecting Anti-Israeli and Anti-Semitic Comments on Reddit

Reddit is the home for thousands of forums (`subreddits`) on various subjects that vary from mainstream to niche. Its slogan is "Dive into anything," as hinted, Reddit discussions tend to be relatively deeper than discussions on other social media networks. Another important feature of Reddit is the participants' anonymity. Each participant (`Redditor`) chooses their username at signup. This feature also leads to more freedom when it comes to expressing extreme and sometimes socially unacceptable views.  
On multiple occasions, Reddit and its CEO were criticized for the extant of hatred spread in some of the communities untouched or regulated, leading to a toxic echo chamber. One prominent victim of such a dynamic is the Jewish/Israeli community, which experienced a surge in anti-semitic comments following the October 7th events in Israel.  

In this project, I try to automate the detection of anti-Israeli and antisemitic comments in the largest news subreddits. I do it by using Reddit's historical data. I start by manually label a golden set. Then, I explored automated labeling schemes and converged into using an ensemble of such methods. I explored the model space as well, and found out that `roberta-base` performed the best. Asking LLMs whether a comment is antisemitic, anti-Israeli or not didn't work out and the model struggled in sarcastic, indirect or unexplicit negative comments. Using LLM for data augmentation ([Balkus and Yan](https://arxiv.org/abs/2205.10981)) helped to achieve a better model and increased the generalization. Id didn't perform well on classification (none of GPT4, GPT3.5, LLaMa 7B and Zephyr 7B alpha).  
In general, the classification of comments on Reddit was challenging, as many of them aren't trivial. Table 1 presents some titles and comments. While all of them are anti-Israeli, they were labeled as `Neutral` by `Zephyr 7B alpha`. Other LLMs had a similar problem, that is why I couldn't relay solely on it for labeling or classiffication of comments.  

| Subreddit | Title | Comment | LLM Label | 
| --------- | ----- | ------- | --------- |
| /r/WorldNews | BBC News -Israel impose tax sanction against Palestinian Authority in retaliation for signing a number of international treaties | There are lots of opponents of BDS (boycotts, divestment and sanctions) which are directed at Israel, as they say it is unhelpful to arriving at a solution. Well the Israeli government just directed sanctions against the westbank (while it boycotts/blockades gaza). | Neutral |
| /r/WorldNews | BBC News -Israel impose tax sanction against Palestinian Authority in retaliation for signing a number of international treaties | This is how Israel reacts to peaceful diplomatic efforts on the part of Palestinians. | Neutral |
| /r/WorldNews | BBC News -Israel impose tax sanction against Palestinian Authority in retaliation for signing a number of international treaties | Israel... just 50 years ago you were in a similar place to the PA.. scattered, beaten, persecuted, stateless. One would think that out of all peoples, Israel would know not to oppress someone. 'Those who do not learn from history are condemned to repeat it'... couldn't be a truer statement in this conflict. | Neutral |
| /r/WorldNews | BBC News -Israel impose tax sanction against Palestinian Authority in retaliation for signing a number of international treaties | Stay classy Israel | Pro Israeli |
| /r/WorldNews | "State of Palestine" allowed to join Geneva Conventions | You can dropped the double quote marks, they are reserved for "israel". | Neutral |
| /r/WorldNews | UN to summon Kim Jong Un before international court; China says it will veto any action against its ally and may block the summoning. | The US vetoes international legal action against its "ally" Israel all the time of course, so China has a long way to catch up in the defying-international-law stakes. | Neutral |
| /r/WorldNews | Israel destroys aid projects in West Bank to make room for settlements | We see Russians acting as victims, but they are the ones taking others's land... We see Israelites acting as victims, but they are the ones taking others land... Are we in a age where the victims are the thieves, or the thieves are the victims? | Neutral |
| /r/WorldNews | France‚Äôs highest appeal court has ordered the country‚Äôs major Jewish organization to pay damages for falsely claiming that a charity supporting Palestinians collected money for Hamas. | I'm sure the Israel Internet cronies will be on this in a minuet, but I'm glad that Palatine aid groups are starting to fight back from the tide of Israeli groups always comparing them to terrorist. | Neutral |
| /r/WorldNews | U.S. officials angry: Israel doesn‚Äôt back stance on Russia | Israel cares about itself, and no one else at all. It doesn't feel indebted to the U.S., why is anyone surprised? Time to change our stance towards the Palestine-Israel conflict. | Neutral |

## Modeling
The modeling procedure is depicted in the following figure. As can be seen in the above table, looking at comments alone can be misleading. Therefore, I take the title and the comment and combine them using the following rule `f'{title} ; {comment}`. Both in `/r/worldnews` and `/r/news`, the submission's title is the title of the article being referred to. It has no additional content. Therefore, comments were made by referring to the title or the referred news article.  
BERT based models can handle sequences of up to a length of `512` tokens. In some cases the combinations of title and a comment surpass this window. In these cases I look only at the first part of the comment. A future work could extend the treatment for long sequences by pooling predictions for different parts, or by any other method. Nevertheless, it is important to note that the mass of title-comment pairs that exceed the 512 token limitation is marginal (see [here](https://github.com/DavidHarar/Reddit/blob/main/notebooks/Analysis.ipynb) under *Lengths*).  

![Modeling](https://github.com/DavidHarar/Reddit/blob/main/plots/modeling.png)
For LLM I used the [Zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha). For classification, I used [Roberta](https://huggingface.co/roberta-base). Label-spreading was done using the `[CLS]` token, using a `KNN` kernel. A voting ensemble was used between the LLM and the label spreading because neither of the methods was good enough. After getting to the combined labeled data, I upscaled my data by asking the LLM to rephrase comments. I do it only once. The final version is of a fine-tuning using the labeled data, after I continue the pre-training, using the entire set of unlabeled data (it improves precision mostly on the right-most side of the predicted scores, in comparison with the just-fine-tuned model). Fine-tuning was done by training the pre-trained model for three epochs on the labeled data.  

# Results
The following plot shows the results on the test set. In general it seems that we have a relatively good precision around the rightmost part of the scores distribution. Since for me **the cost of missclassifying a comment as antisemitic/anti Israeli when it is not, I choose a very high threshold of 0.9, even at a cost of missing some of the true antisemitic/anti Israeli comments on Reddit**.  
![KPIs](https://github.com/DavidHarar/Reddit/blob/main/plots/KPIs.png)

In general it seems that the fine-tuned model was able to learn some of the meanings of the comments it processed. In the next plot I depict the UMAP projection of the `[CLS]` tokens taken from the fine-tuned `roberta-base` model. We can see that it learned something related to the concept which we try to teach it by the fact that it split the points into two distinct clusters, and that the color of vast majority of the points (commets) in each of them is related to one label.  
![UMAP](https://github.com/DavidHarar/Reddit/blob/main/plots/umap.png)


# Some Correctly Detected Comments
As mentioned above, a large portion of the comments are pretty complex. Also, there is a possible data drift. While a lot of anti-Israeli comments in the training data were considered trivial (e.g. "Fuck Israel"), the negative comments about Israel tend to have more complex structures. The following table presents comments that were detected correctly. Nevertheless, even with a choice of a very high threshold (`.9`), it seems that some positive/unrelated comments end up being detected.  
For more examples of detected comments, visit `notebooks/detect_comments_using_praw.ipynb`.


| Subreddit | Title | Comment |
| --------- | ----- | ------- |
| /r/worldnews | Released Thai hostage says Israelis held with him were beaten with electric cables|What did Israel expect would happen when they try and colonise a country? Eventually they are gonna fight back |
| /r/worldnews | Released Thai hostage says Israelis held with him were beaten with electric cables|as horrible as it is, Palestinians are treated horribly in Israeli prisons as well. Violence begets violence |
| /r/worldnews | Released Thai hostage says Israelis held with him were beaten with electric cables|Good to see they got the talking points deliver by mossad |
| /r/worldnews | Erdogan tells UN chief Israel must be tried in international courts for Gaza crimes|I don’t think fighting the Jewish State in a court of law is gonna go the way this guy thinks it will. |
| /r/worldnews | Erdogan tells UN chief Israel must be tried in international courts for Gaza crimes|Yeah... Israel should stand trial for what it is doing in Gaza, No question about it. I don't care what they believe they are entitled to*they are NOT allowed to kill and displace 2million + people from their homes and likely cause Many thousands of peoples death in doing so In excess of them BOMBING their god damn houses, hospitals and businesses.** |
| /r/worldnews | Israel's finance minister defends settlement funds in budget row|These POS people need to be jailed.. we need to fight this cancer. So much damage is being done for years it's horrible... |
| /r/worldnews | Israel's finance minister defends settlement funds in budget row|Israeli policy towards settlers couldn’t be more inflammatory, antagonistic, and radicalizing for Palestinians who have to suffer them. |
| /r/worldnews | Freed Israeli hostage describes deteriorating conditions while being held by Hamas|My condition deteriorated as the IDF treated me like a one of nearly 6 million Palestinian hostages in occupied Bantustans, can you imagine? |
| /r/worldnews | Freed Israeli hostage describes deteriorating conditions while being held by Hamas|Oh she was starving? It is almost like there Israel cut off the food and water in Gaza 🤯🤯 |


## Limitations and Further Work
I made accommodations on multiple occasions to see the project through.
- Data: I used old Reddit data using the Pushlift Data Dumps. Each subreddit includes two files, one for submissions and another for comments made for each sumbission, and for each comment. I didn't address comments to comments and basically trained a model using only comments made directly to the submission itself. 
- Model: As mentioned above, I didn't solve the issue of long strings. Further work would be able to solve it. 
- Labeled data: I manually labeled only 3k observations. In order to have more robust data, one should label more examples. 

## Downvote and Legal
In Reddit, there are two main social coins. One is `karma` and another is `awards`. The latter is basically digital prizes given between Redditors to express appreciation for meaningful comments. Most of these awards cost money, and this project ignores them entirely. The other coin, `karma`, is basically the balance of upvotes and downvotes of the comments a Redditor has made. Being downvoted for a hateful comment will inflict on the user's karma.  

Reddit API allows for downvotes and upvotes of comments and submissions directly from the API. Nevertheless, the usage of automated voting may be considered a violation of their policies, and hence **I do not recommend it, and any change to the provided code is on the modifier's responsibility**. For this reason, in this project, I only reveal comments that are suspected of being antisemitic or anti-Israeli and take no action further than that.  
Regarding the use of Reddit'd Data in a machine learning model (see [3.2 Restrictions](https://www.redditinc.com/policies/data-api-terms?fbclid=IwAR3H0FWGTMVo1W0hJJhkBD31knspw36N5DvbO5RiBfz1bIl8lXk_tZLtKnE), Data API Terms), it is said that data that atchieved through Reddit's API won't be used to train ML models effective June 19, 2023. The data used in this project was taken from [Reddit Pushlift](https://www.reddit.com/r/pushshift/), a community that aggregates Reddit's old data from when it was perfectly legal to be used for training ML models.  


## How to get your Reddit API Key
In order to get such report, one has to have a Reddit API key. [This](https://www.reddit.com/wiki/api/) is a guide for how to get one. For convenience I summarize the steps below:
1. Read [Reddit’s Developer Terms](https://www.redditinc.com/policies/developer-terms) and [Data API Terms](https://www.redditinc.com/policies/data-api-terms). In section 4 it is specified that one should not use Reddit's data to train ML models. The restriciton is effective June 2023, while the data that was used to train the model is 10 years old. In the current project, no data is being collected nor being used for retraining the provided model.  
2. [Register](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164). Here you have to specify your interest. 
3. Get your personal API key from [here](https://old.reddit.com/prefs/apps/).