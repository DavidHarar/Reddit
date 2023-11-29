# Detecting Anti-Israli and Anti-Semitic Comments on Reddit

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
For LLM I used the [Zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha). For classification I used [Roberta](https://huggingface.co/roberta-base). Label-spreading was done using the `[CLS]` token, using a `KNN` kernel. Voting ensemble was used between the LLM and the label spreading because neither of the method was good enough. After getting to the combined labeled data, I upscaled my data by asking the LLM to rephrase comments. I do it only once. The final version is of a fine-tuning using the labeled data, after I continue the pre-training, using the entire set of unlabeled data.  


## Downvote
In Reddit there are two main social coins. One is `karma` and another is `awards`. The latter is basically diginal prizes given between redditors to express upreciation for meaningfull comments. Most of these awards cost money, and this project ignores them entirely. The other coin, `karma`, is basically the balance of upvotes and downvotes of the comments a redittor has made. Being downvoted for a hateful comment will inflict on the user's karma.  

Reddit allows the usage of automated tools when it comes to classify and vote on comments. It doesn't allow for fake accounts. In this project we offer a tool to automate the search and downvote of anti-semitic/Israeli comments, given a Reddit developer key. We do not offer or suggesting to create fake accounts by any means.  

## How to get your Reddit Developer Key

