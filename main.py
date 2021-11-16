import pandas as pd
import re
import random as rd
import math
import sys
pd.options.mode.chained_assignment = None

class TweetCluster:
    def __init__(self, url, verbose=True, very_verbose=False):
        self.data_url = url
        self.tweet_df = pd.read_csv(url, '|', encoding= 'unicode_escape')
        self.tweet_df.columns = ['ID', 'Time_Stamp', 'Tweet']
        self.backup_df = pd.DataFrame()
        self.k = 0
        self.book_keep = []

        # Previous iteration of the centroid.
        self.prev_centroid = []

        # Will act as the "New centroid" or the most updated one.
        self.centroid = []

        self.map = {}
        self.clusters = {}
        self.verbose = verbose
        self.very_verbose = very_verbose

    def pre_process_tweets(self):
        self.backup_df = self.tweet_df['Tweet']
        for i in range(0, len(self.backup_df)):
            # The RE Matches all of the requirements.
            self.backup_df.iloc[i] = re.sub(r"\.?[@,#]\S+\s|\s[@,#]\S+|\s?http[s]?:[/]{2,2}\S+|\s?www\S+",
                                            "", self.backup_df.iloc[i])
            self.backup_df.iloc[i] = self.backup_df.iloc[i].lower()
        self.tweet_df = self.backup_df

    def __get_distance(self, tweet1, tweet2):
        intersection = set(tweet1).intersection(tweet2)

        union = set().union(tweet1, tweet2)

        # Jaccard Distance equation
        return 1 - (len(intersection) / len(union))

    # Call to reset before k_means call.
    # Prevents having to make another class and having to preprocess again
    # Thus, speeds it up a little bit more.
    def reset(self):
        if self.verbose:
            print("Resetting Twitter Cluster Class")
        self.tweet_df = self.backup_df
        self.tweet_df.reset_index(drop=True, inplace=True)
        self.prev_centroid = []
        self.centroid = []
        self.map = {}
        self.clusters = {}

    def __populate_cluster(self):
        # Need to reset the clusters every iteration.
        self.clusters = {}
        if(self.verbose):
            print("Populating the clusters with tweets.")
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

            # setdefault is a cool function because it will check if the key given exists.
            # If it does not exists, then add the key:value relationship give, if it exists then return its value.
            self.clusters.setdefault(cluster_index, []).append([self.tweet_df.iloc[i]])
            if self.very_verbose:
                print("Tweet number: "+str(i)+", '"+self.tweet_df.iloc[i]+"' is assigned to cluster number: "+str(cluster_index))


            # Here I will add tweet distance from its centroid for the future to easily compute SSE
            last_tweet = len(self.clusters.setdefault(cluster_index, [])) - 1
            self.clusters.setdefault(cluster_index, [])[last_tweet].append(min_distance)
            # print("CLUSTER: "+str(self.clusters))

    def __is_converged(self):
        if len(self.prev_centroid) != len(self.centroid):
            return False

        for i in range(len(self.centroid)):
            if self.centroid[i] != self.prev_centroid[i]:
                return False
        return True

    def __update_centroids(self):
        if self.very_verbose:
            print("Updating Centroids: " + str(self.centroid))
        elif self.verbose:
            print("Updating Centroids")

        # We saved the centroids in the self.prev_centroid, we now do the update rule
        self.centroid = []

        # Goal is to iterate each cluster and get the average distance sum and then change the centroid to the tweet
        # that has the lowest average distance.
        # The for loop will scan in the order the clusters are initialized. This version of the for loop
        # Guarentees that no clusters that are empty will be ran.
        for cluster in self.clusters:
            if self.verbose:
                print("Scanning Cluster "+str(cluster))
            min_distance_avg = math.inf
            centroid_index = -1

            # To speed of the process I apply some dynamic programming with this list
            min_dist_dynamic = []
            for t1 in range(len(self.clusters[cluster])):
                min_dist_dynamic.append([])
                distance_sum = 0

                # This is when we get the distance sum for every other tweet for t1
                for t2 in range(len(self.clusters[cluster])):
                    if t1 != t2:
                        if t2 < t1:
                            temp_dist = min_dist_dynamic[t2][t1]
                        else:
                            temp_dist = self.__get_distance(self.clusters[cluster][t1][0], self.clusters[cluster][t2][0])
                        distance_sum += temp_dist
                        min_dist_dynamic[t1].append(temp_dist)
                    else:
                        min_dist_dynamic[t1].append(0)

                average = distance_sum/len(self.clusters[cluster])
                if average < min_distance_avg:
                    min_distance_avg = average
                    centroid_index = t1
                    if self.very_verbose:
                        print("Minimum average update cluster: "+str(cluster)+" and Tweet #"+str(t1)+"Tweet: "
                              +str(self.clusters[cluster][centroid_index][0]))
            self.centroid.append(self.clusters[cluster][centroid_index][0])

    def __book_keep(self, SSE):
        self.book_keep.append({'k': self.k, 'SSE': SSE,
                               'cluster size':[str(i)+": "+str(len(self.clusters[i])) for i in self.clusters]
                               })
        if self.very_verbose:
            print("Added entry into Book Keeping: "+str(self.book_keep))
        elif self.verbose:
            print("Added entry into Book Keeping")

    def tabulate_output(self):
        print("Printing to output.txt...")
        file = open("output.txt", "w")
        file.write("__________________________________________________________\n")
        file.write("|  k value  |        SSE        |  Size of each cluster  |\n")
        file.write("|___________|___________________|________________________|\n")
        for row in self.book_keep:
            first_cluster = row['cluster size'][0]

            file.write("|{:^11}|{:^19}|{:^24}|\n".format(row['k'], '%.5f'%row['SSE'], first_cluster))
            for cluster in row['cluster size']:
                if first_cluster != cluster:
                    file.write("|{:^11}|{:^19}|{:^24}|\n".format("", "", cluster))
            file.write("|___________|___________________|________________________|\n")
        file.close()
        print("Finished printing.")

    def k_means(self, k=5, max_iterations=50):
        self.k = k
        count = 0
        while count < k:
            random_tweet = rd.randint(0, len(self.tweet_df)-1)
            if random_tweet not in self.map:
                self.map[random_tweet] = True
                self.centroid.append(self.tweet_df.iloc[random_tweet])
                count += 1

        iterations = 0
        while (not self.__is_converged()) and (iterations < max_iterations):
            if self.verbose:
                print("Iteration: "+str(iterations))
            self.__populate_cluster()

            self.prev_centroid = self.centroid
            self.__update_centroids()
            iterations += 1

        if iterations == max_iterations:
            print("K means did not converge. The iteration count hit the max.")
        else:
            print("K means Converged on iteration: "+str(iterations))

        # Calculate Sum Square Error

        # Instead of keeping SSE as an attribute I chose to return it since I think it makes more sense
        # for the user to have access to the SSE by calling k_means rather than just getting the attribute.
        SSE = 0
        for cluster in self.clusters:
            # t stands for tweet index.
            # For loop inherently checks if cluster is empty
            for tweet in self.clusters[cluster]:
                # here can can utilize some more dynamic programming
                # We stored the distance with the tweets in the clusters for n^2 computation time
                # Without having to do intersections or unions
                SSE += (tweet[1]**2)

        self.__book_keep(SSE)
        self.reset()
        return SSE

if len(sys.argv) == 2:
    if sys.argv[1] == "vv" or sys.argv[1] == "VV":
        very_verbose = True
        verbose = True
    elif sys.argv[1] == "v" or sys.argv[1] == "VV":
        verbose = True
        very_verbose = False
    else:
        print("There was a incorrect argument given\nUse vv/VV for very verbose\nUse v/V for regular verbose")
        exit()
else:
    verbose = False
    very_verbose = False

cluster = TweetCluster("https://raw.githubusercontent.com/caige13/Twitter_Clustering_Data/main/foxnewshealth.txt", verbose, very_verbose)
#cluster = TweetCluster("./Health-News-Tweets/Health-Tweets/foxnewshealth.txt", True)
cluster.pre_process_tweets()

if very_verbose:
    # 1000 on VV is way too much.
    k_values = [2, 5, 10, 20, 50, 100]
else:
    k_values = [2, 5, 10, 20, 50, 100, 1000]
for k in k_values:
    print("\n\nSTARTING: k="+str(k))
    SSE = cluster.k_means(k)
    print("\nFOR k = " + str(k) + " SSE is: " + str(SSE))
cluster.tabulate_output()
