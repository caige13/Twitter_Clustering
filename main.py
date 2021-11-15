import pandas as pd
import re
import random as rd
import math
import sys

class TweetCluster:
    def __init__(self, url, verbose = True):
        self.data_url = url
        self.tweet_df = pd.read_csv(url, '|', encoding= 'unicode_escape')
        self.tweet_df.columns = ['ID', 'Time_Stamp', 'Tweet']

        # Previous iteration of the centroid.
        self.prev_centroid = []

        # Will act as the "New centroid" or the most updated one.
        self.centroid = []

        self.map = {}
        self.clusters = {}
        self.verbose = verbose

    def pre_process_tweets(self):
        temp_df = self.tweet_df['Tweet']
        for i in range(0, len(temp_df)):
            temp_df.iloc[i] = re.sub(r"\.?[@,#]\S+\s|\s[@,#]\S+|http[s]?:[/]{2,2}\S+|www\S+", "", temp_df.iloc[i])
            temp_df.iloc[i] = temp_df.iloc[i].lower()
        self.tweet_df = temp_df

    def k_means(self, k=5, max_iterations=50):
        count = 0
        while count < k:
            random_tweet = rd.randint(0, len(self.tweet_df))
            if random_tweet not in self.map:
                map[random_tweet] = True
                self.centroid.append(self.tweet_df.iloc[random_tweet])
                count += 1

        iterations = 0
        while not self.__is_converged and iterations < max_iterations:
            if self.verbose:
                print("Iteration: "+str(iterations), end="\t")
            self.__assign_cluster()

    def __get_distance(self, tweet1, tweet2):
        intersection = set(tweet1).intersection(tweet2)

        union = set().union(tweet1, tweet2)

        # Jaccard Distance equation
        return 1 - (len(intersection) / len(union))

    def __assign_cluster(self):
        for i in range(len(self.tweet_df)):
            min_distance = math.inf
            cluster_index = -1
            for j in range(len(self.centroid)):
                distance = self.__get_distance(self.centroid[j], self.tweet_df.iloc[i])

                if self.centroid[j] == self.tweet_df.iloc[i]:
                    cluster_index = j
                    min_distance = 0
                    break
                if distance < min_distance:
                    cluster_index = j
                    min_distance = distance

            if min_distance == 1:
                cluster_index = rd.randint(0, len(self.centroid)-1)


    def __is_converged(self):
        if len(self.centroid) == 0:
            return True
        if len(self.prev_centroid) != len(self.centroid):
            return False

        for i in range(len(self.centroid)):
            if self.centroid[i] != self.prev_centroid[i]:
                return False
        return True


verbose = sys.argv[1]
if verbose == "v" or verbose == "V":
    verbose = True
else:
    verbose = False

#cluster = TweetCluster("https://raw.githubusercontent.com/caige13/Twitter_Clustering_Data/main/foxnewshealth.txt")
cluster = TweetCluster("./Health-News-Tweets/Health-Tweets/foxnewshealth.txt", verbose)
cluster.pre_process_tweets()

