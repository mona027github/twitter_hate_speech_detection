import tweepy

# initialize api instance
consumer_key = '5bZAf2lKwTBu6dVZONz5tSZji'
consumer_secret='eDVqN7vZbEni0Ho4dLc5drP6vMdwPJIHrk4WRGcYL0MyV9YzG0'
access_token = '893669907304292353-8ZgFZNnBemESKRiaBb9lHyRPN3LV7a8'
access_token_secret = 'UoL2ojUPP2q0mYvUeKCqFTmGs9bn8TsTlUSFtolLyeKhB'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

import csv
csvFile = open('result.csv','a')
csvWriter = csv.writer(csvFile)

api = tweepy.API(auth, wait_on_rate_limit=True)
for tweet in tweepy.Cursor(api.search, q="jihadism", lang="en").items(1000):
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

csvFile.close()


import codecs
import nltk
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

with codecs.open('result.csv','r') as csvfile:
    for tweet in csvfile:

        tokenized_tweets = tknzr.tokenize(tweet)

        with open('result1.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(tokenized_tweets)

filename = 'result1.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
# split based on words only
# import os
# os.remove('hate.csv')
# os.remove('result1.csv')
import re
alpha_num_values = re.split(r'\W+', text)


filtered_tokens = [x for x in alpha_num_values if not any(c.isdigit() for c in x)]
#print(filtered_tokens)

lemma = nltk.wordnet.WordNetLemmatizer()
stemmed_words = [lemma.lemmatize(word) for word in filtered_tokens]
#print(stemmed_words)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in stemmed_words if not w in stop_words]
# print(words)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()
for word in words:
  vs = analyzer.polarity_scores(word)

  # print("{:-<65} {}".format(word, str(vs)))


    # Create a SentimentIntensityAnalyzer object.
sid_obj = SentimentIntensityAnalyzer()
negative_list = []


for word in words:
    sentiment_dict = sid_obj.polarity_scores(word)

    if sentiment_dict['compound'] <= - 0.05:
        # print("Overall sentiment dictionary is : ", sentiment_dict)
        #
        #
        # print("word was rated as ", sentiment_dict['neg'] * 100, "% Negative")
        negative_list.append(word)
# print(negative_list)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
from sklearn.feature_extraction.text import TfidfVectorizer
# create the transform
# df =pd.read_csv('result.csv')
vectorizer = TfidfVectorizer()
# tokenize and build vocab

vectorizer.fit(negative_list)
# summarize
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# encode document
vector = vectorizer.transform(negative_list)

# summarize encoded vector
vector.shape

tf_idf_array =vector.toarray

# arr2D = vectorizer.idf_

#
# with open('hate.csv', 'a') as output:
#     out = csv.writer(output)
#     out.writerows(map(lambda x: [x], vectorizer.idf_))
import pandas as pd

pd.DataFrame((tf_idf_array), columns=vectorizer.get_feature_names()).head()

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score
#
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(negative_list)
#
# true_k = 3
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)
#
# print("Top terms per cluster:")
#
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :6]:
#
#         print(' %s' % terms[ind]),
#     print
#
# print("\n")
# print("Prediction")

# Y = vectorizer.transform(["chrome browser to open."])
# prediction = model.predict(Y)
# print(prediction)
#
# Y = vectorizer.transform([negative_list])
# prediction = model.predict(Y)
# print(prediction)


























# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# df = pd.read_csv('result.csv' ,names=['V1','V2'])
# print(df.shape)
# print(df.head())
# f1=df['V1'].values
# f2=df['V2'].values
# X = np.array(list(zip(f1,f2)))
# k=2
# C_x = np.random.randint(0, np.max(X)-20, size=k)
# C_y = np.random.randint(0, np.max(X)-20, size=k)
# C=  np.array(list(zip(C_x,C_y)), dtype=np.float(32))
#
# plt.scatter(f1,f2, c='black',s=20)
# plt.scatter(C_x,C_y, marker="*", c='green',s=200)
# plt.xlabel('V1')
# plt.ylabel('V2')
# plt.show()

class Kmeans:
    """ K Means Clustering

    Parameters
    -----------
        k: int , number of clusters

        seed: int, will be randomly set if None

        max_iter: int, number of iterations to run algorithm, default: 200

    Attributes
    -----------
       centroids: array, k, number_features

       cluster_labels: label for each data point

    """

    def _init_(self, k, seed=None, max_iter=200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    def initialise_centroids(self, arr2D):
        """Randomly Initialise Centroids

        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        centroids: array of k centroids chosen as random data points
        """

        initial_centroids = np.random.permutation(arr2D.shape[0])[:self.k]
        self.centroids =arr2D[initial_centroids]

        return self.centroids

    def assign_clusters(self, arr2D):
        """Compute distance of data from clusters and assign data point
           to closest cluster.

        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster

        """

        if arr2D.ndim == 1:
            arr2D = arr2D.reshape(-1, 1)

        dist_to_centroid = pairwise_distances(arr2D, self.centroids, metric='euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, arr2D):
        """Computes average of all data points in cluster and
           assigns new centroids as average of data points

        Parameters
        -----------
        data: array or matrix, number_rows, number_features

        Returns
        -----------
        centroids: array, k, number_features
        """

        self.centroids = np.array([arr2D[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])

        return self.centroids

    def predict(self,arr2D):
        """Predict which cluster data point belongs to

        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster
        """

        return self.assign_clusters(arr2D)

    def fit_kmeans(self, arr2D):
        """
        This function contains the main loop to fit the algorithm
        Implements initialise centroids and update_centroids
        according to max_iter
        -----------------------

        Returns
        -------
        instance of kmeans class

        """
        self.centroids = self.initialise_centroids(arr2D)

        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(arr2D)
            self.centroids = self.update_centroids(arr2D)
            if iter % 100 == 0:
                print("Running Model Iteration %d " % iter)
        print("Model finished running")
        return self

    # number_clusters = range(1, 7)
    #
    # kmeans = [KMeans(n_clusters=i, max_iter=600) for i in number_clusters]
    # kmeans
    #
    # score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]
    # score
    #
    # plt.plot(number_clusters, score)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Score')
    # plt.title('Elbow Method')
    # plt.show()

sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)
Y_sklearn=  np.array(list(zip(Y_sklearn)), dtype=np.float(32))

test_e = Kmeans(3, 1, 600)
fitted = test_e.fit_kmeans(Y_sklearn)
predicted_values = test_e.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis')

centers = fitted.centroids
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);







