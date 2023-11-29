# Detecting Anti-Israli and Anti-Semitic Comments on Reddit

Reddit is the home for thousands of forums (`subreddits`) on various subjects that vary from mainstream to niche. It's slogan is "Dive into anything", and a as hinted, discussions on Reddit tend to be, relatively deeper than discussion on other social media networks. Another important feature of Reddit is the participants' anonimity. Each participant (`Redditor`) chooses their username at signup. This feature also lead to the more freedom when if comes to expressing extream and sometimes socialy unexeptable views.  
On multiple occations Reddit and it's CEO were critisized for the extant of hatered spread in some of the communities untouched or regulated, leading to a toxic echo chamber. One obvious victim of such dinamic is the jewish/Israeli community, that experienced a surge in anti-semitic comments following the October 7th events in Israel.  

In this project I build a classifier to detect anti-Israeli and anti-semitic comments. I then combine the model with Reddit's API, `PRAW`, in order to search through recent comments and downvote them for being anti-Israeli or anti-semitic.  

## Modeling



## Downvote
In Reddit there are two main social coins. One is `karma` and another is `awards`. The latter is basically diginal prizes given between redditors to express upreciation for meaningfull comments. Most of these awards cost money, and this project ignores them entirely. The other coin, `karma`, is basically the balance of upvotes and downvotes of the comments a redittor has made. Being downvoted for a hateful comment will inflict on the user's karma.  

Reddit allows the usage of automated tools when it comes to classify and vote on comments. It doesn't allow for fake accounts. In this project we offer a tool to automate the search and downvote of anti-semitic/Israeli comments, given a Reddit developer key. We do not offer or suggesting to create fake accounts by any means.  

## How to a Reddit Developer Key