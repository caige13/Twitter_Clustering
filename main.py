import pandas as pd
import re

class TweetCluster:
    def __init__(self, url):
        self.data_url = url
        self.tweet_df = pd.read_csv(url, '|', encoding= 'unicode_escape')
        self.tweet_df.columns= ['ID', 'Time_Stamp', 'Tweet']

    def pre_process_tweets(self):
        temp_df = self.tweet_df['Tweet']
        for i in range(0, len(temp_df)):
            temp_df.iloc[i] = re.sub(r"[@,#]\S+\s|\s[@,#]\S+|http[s]?:[/]{2,2}\S+", "", temp_df.iloc[i])
            print(temp_df.iloc[i])

#cluster = TweetCluster("https://raw.githubusercontent.com/caige13/Twitter_Clustering_Data/main/foxnewshealth.txt")
cluster = TweetCluster("./Health-News-Tweets/Health-Tweets/foxnewshealth.txt")
cluster.pre_process_tweets()

