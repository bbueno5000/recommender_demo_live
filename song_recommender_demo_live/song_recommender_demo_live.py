"""
DOCSTRING
"""
import csv
import math
import numpy
import pandas
import pylab
import random
import scipy
import sklearn
import sparsesvd
import time

class ItemSimilarityRecommenderPy():
    """
    Class for Item similarity based Recommender System model.
    """
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    def construct_cooccurence_matrix(self, user_songs, all_songs):
        """
        DOCSTRING
        """
        user_songs_users = list()
        for i in range(len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
        cooccurence_matrix = numpy.matrix(numpy.zeros(shape=(len(user_songs), len(all_songs))), float)
        for i in range(len(all_songs)):
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            for j in range(0,len(user_songs)):
                users_j = user_songs_users[j]
                users_intersection = users_i.intersection(users_j)
                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0
        return cooccurence_matrix

    def create(self, train_data, user_id, item_id):
        """
        Create the item similarity based recommender system model.
        """
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        """
        Use the cooccurence matrix to make top recommendations.
        """
        print('Non zero values in cooccurence_matrix:{}'.format(numpy.count_nonzero(cooccurence_matrix)))
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = numpy.array(user_sim_scores)[0].tolist()
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)
        columns = ['user_id', 'song', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)
        # fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(len(sort_index)):
            if ~numpy.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank+1
        # handle the case where there are no recommendations
        if df.shape[0] == 0:
            print('The current user has no songs for training the item similarity based recommendation model.')
            return -1
        else:
            return df

    def get_all_items_train_data(self):
        """
        Get unique items (songs) in the training data.
        """
        all_items = list(self.train_data[self.item_id].unique())
        return all_items

    def get_item_users(self, item):
        """
        Get unique users for a given item (song).
        """
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users

    def get_similar_items(self, item_list):
        """
        Get similar items to given items.
        """
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print('no. of unique songs in the training set:{}'.format(len(all_songs)))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations

    def get_user_items(self, user):
        """
        Get unique items (songs) corresponding to a given user.
        """
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items

    def recommend(self, user):
        """
        Use the item similarity based recommender system model to make recommendations.
        """
        user_songs = self.get_user_items(user)
        print('No. of unique songs for the user: {}'.format(len(user_songs)))
        all_songs = self.get_all_items_train_data()
        print('no. of unique songs in the training set:{}'.format(len(all_songs)))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations

class PrecisionRecallCalculator:
    """
    Class to calculate precision and recall.
    """
    def __init__(self, test_data, train_data, pm, is_model):
        self.ism_training_dict = dict()
        self.model1 = pm
        self.model2 = is_model
        self.pm_training_dict = dict()
        self.test_data = test_data
        self.test_dict = dict()
        self.train_data = train_data
        self.user_test_sample = None

    def calculate_measures(self, percentage):
        """
        A wrapper method to calculate all the evaluation measures.
        """
        self.create_user_test_sample(percentage)
        self.get_test_sample_recommendations()
        return self.calculate_precision_recall()
        
    def calculate_precision_recall(self):
        """
        Method to calculate the precision and recall measures.
        """
        cutoff_list = list(range(1, 11))
        ism_avg_precision_list = list()
        ism_avg_recall_list = list()
        pm_avg_precision_list = list()
        pm_avg_recall_list = list()
        num_users_sample = len(self.users_test_sample)
        for N in cutoff_list:
            ism_sum_precision = 0
            ism_sum_recall = 0
            pm_sum_precision = 0
            pm_sum_recall = 0
            ism_avg_precision = 0
            ism_avg_recall = 0
            pm_avg_precision = 0
            pm_avg_recall = 0
            for user_id in self.users_test_sample:
                ism_hitset = self.test_dict[user_id].intersection(set(
                    self.ism_training_dict[user_id][0:N]))
                pm_hitset = self.test_dict[user_id].intersection(set(
                    self.pm_training_dict[user_id][0:N]))
                testset = self.test_dict[user_id]
                pm_sum_precision += float(len(pm_hitset)) / float(N)
                pm_sum_recall += float(len(pm_hitset)) / float(len(testset))
                ism_sum_precision += float(len(ism_hitset)) / float(len(testset))
                ism_sum_recall += float(len(ism_hitset)) / float(N)
            pm_avg_precision = pm_sum_precision / float(num_users_sample)
            pm_avg_recall = pm_sum_recall / float(num_users_sample)
            ism_avg_precision = ism_sum_precision / float(num_users_sample)
            ism_avg_recall = ism_sum_recall / float(num_users_sample)
            ism_avg_precision_list.append(ism_avg_precision)
            ism_avg_recall_list.append(ism_avg_recall)
            pm_avg_precision_list.append(pm_avg_precision)
            pm_avg_recall_list.append(pm_avg_recall)
        return (
            pm_avg_precision_list, pm_avg_recall_list,
            ism_avg_precision_list, ism_avg_recall_list)

    def create_user_test_sample(self, percentage):
        """
        Create a test sample of users for use in calculating precision and recall.
        """
        # find users common between training and test set
        users_test_and_training = list(set(self.test_data['user_id'].unique()).intersection(
            set(self.train_data['user_id'].unique())))
        print('Length of user_test_and_training:{}'.format(len(users_test_and_training)))
        # take only random user_sample of users for evaluations
        self.users_test_sample = self.remove_percentage(users_test_and_training, percentage)
        print('Length of user sample:{}'.format(len(self.users_test_sample)))
        
    def get_test_sample_recommendations(self):
        """
        Method to generate recommendations for users in the user test sample.
        """
        # for these test_sample users, get top 10 recommendations from training set
        for user_id in self.users_test_sample:
            # get items for user_id from item similarity model
            print('Getting recommendations for user:{}'.format(user_id))
            user_sim_items = self.model2.recommend(user_id)
            self.ism_training_dict[user_id] = list(user_sim_items['song'])
            # get items for user_id from popularity model
            user_sim_items = self.model1.recommend(user_id)
            self.pm_training_dict[user_id] = list(user_sim_items['song'])
            # get items for user_id from test_data
            test_data_user = self.test_data[self.test_data['user_id'] == user_id]
            self.test_dict[user_id] = set(test_data_user['song'].unique())

    def remove_percentage(self, list_a, percentage):
        """
        Method to return random percentage of values from a list.
        """
        k = int(len(list_a) * percentage)
        random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
        return new_list

class PopularityRecommenderPy():
    """
    Class for Popularity based Recommender System model.
    """
    def __init__(self):
        self.item_id = None
        self.popularity_recommendations = None
        self.train_data = None
        self.user_id = None
        
    def create(self, train_data, user_id, item_id):
        """
        Create the popularity based recommender system model.
        """
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):
        """
        Use the popularity based recommender system model to make recommendations.
        """
        user_recommendations = self.popularity_recommendations
        user_recommendations['user_id'] = user_id
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        return user_recommendations

class SongRecommender:
    """
    DOCSTRING
    """
    def __init__(self):
        # constants defining the dimensions of our User Rating Matrix (URM)
        self.max_pid = 4
        self.max_uid = 5

    def __call__(self):
        triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
        songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
        song_df_1 = pandas.read_table(triplets_file,header=None)
        song_df_1.columns = ['user_id', 'song_id', 'listen_count']
        song_df_2 =  pandas.read_csv(songs_metadata_file)
        song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
        song_df.head()
        len(song_df)
        song_df = song_df.head(10000)
        song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
        song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
        grouped_sum = song_grouped['listen_count'].sum()
        song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
        song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
        users = song_df['user_id'].unique()
        len(users)
        # TODO: Fill in the code here
        songs = song_df['song'].unique()
        len(songs)
        train_data, test_data = sklearn.cross_validation.train_test_split(
            song_df, test_size = 0.20, random_state=0)
        print(train_data.head(5))
        pm = PopularityRecommenderPy()
        pm.create(train_data, 'user_id', 'song')
        user_id = users[5]
        pm.recommend(user_id)
        # TODO: Fill in the code here
        user_id = users[8]
        pm.recommend(user_id)
        is_model = ItemSimilarityRecommenderPy()
        is_model.create(train_data, 'user_id', 'song')
        user_id = users[5]
        user_items = is_model.get_user_items(user_id)
        print("Training data songs for the user userid: %s:" % user_id)
        for user_item in user_items:
            print(user_item)
        print("Recommendation process going on:")
        is_model.recommend(user_id)
        user_id = users[7]
        # TODO: Fill in the code here
        user_items = is_model.get_user_items(user_id)
        print("Training data songs for the user userid: %s:" % user_id)
        for user_item in user_items:
            print(user_item)
        print("Recommendation process going on:")
        is_model.recommend(user_id)
        is_model.get_similar_items(['U Smile - Justin Bieber'])
        song = 'Yellow - Coldplay'
        # TODO: Fill in the code here
        is_model.get_similar_items([song])
        start = time.time()
        user_sample = 0.05
        pr = PrecisionRecallCalculator(test_data, train_data, pm, is_model)
        (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)
        end = time.time()
        print(end - start)
        print('Plotting precision recall curves.')
        self.plot_precision_recall(
            pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
            ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
        print("Plotting precision recall curves for a larger subset of data (100,000 rows) (user sample = 0.005).")
        pm_avg_precision_list = sklearn.externals.joblib.load('pm_avg_precision_list_3.pkl')
        pm_avg_recall_list = sklearn.externals.joblib.load('pm_avg_recall_list_3.pkl')
        ism_avg_precision_list = sklearn.externals.joblib.load('ism_avg_precision_list_3.pkl')
        ism_avg_recall_list = sklearn.externals.joblib.load('ism_avg_recall_list_3.pkl')
        print("Plotting precision recall curves.")
        self.plot_precision_recall(
            pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
            ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
        print("Plotting precision recall curves for a larger subset of data (100,000 rows) (user sample = 0.005).")
        pm_avg_precision_list = sklearn.externals.joblib.load('pm_avg_precision_list_2.pkl')
        pm_avg_recall_list = sklearn.externals.joblib.load('pm_avg_recall_list_2.pkl')
        ism_avg_precision_list = sklearn.externals.joblib.load('ism_avg_precision_list_2.pkl')
        ism_avg_recall_list = sklearn.externals.joblib.load('ism_avg_recall_list_2.pkl')
        print("Plotting precision recall curves.")
        self.plot_precision_recall(
            pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
            ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
        K=2 # number of latent factors
        urm = numpy.array([[3, 1, 2, 3],[4, 3, 4, 3],[3, 2, 1, 5], [1, 6, 5, 2], [5, 0,0 , 0]])
        urm = scipy.sparse.csc_matrix(urm, dtype=numpy.float32)
        U, S, Vt = self.compute_svd(urm, K)
        uTest = [4]
        print("User id for whom recommendations are needed: %d" % uTest[0])
        print("Predictied ratings:")
        uTest_recommended_items = self.compute_estimated_ratings(urm, U, S, Vt, uTest, K, True)
        print(uTest_recommended_items)
        print("Matrix Dimensions for U")
        print(U.shape)
        for i in range(0, U.shape[0]):
            plot(U[i,0], U[i,1], marker = "*", label="user"+str(i))
        for j in range(0, Vt.T.shape[0]):
            plot(Vt.T[j,0], Vt.T[j,1], marker = 'd', label="item"+str(j))    
        legend(loc="upper right")
        title('User vectors in the Latent semantic space')
        ylim([-0.7, 0.7])
        xlim([-0.7, 0])
        show()

    def compute_estimated_ratings(self, urm, U, S, Vt, uTest, K, test):
        """
        Compute estimated rating for the test user.
        """
        rightTerm = S*Vt 
        estimatedRatings = numpy.zeros(shape=(self.max_uid, self.max_pid), dtype=numpy.float16)
        for userTest in uTest:
            prod = U[userTest, :]*rightTerm
            #we convert the vector to dense format in order to get the indices 
            #of the movies with the best estimated ratings 
            estimatedRatings[userTest, :] = prod.todense()
            recom = (-estimatedRatings[userTest, :]).argsort()[:250]
        return recom

    def compute_svd(self, urm, K):
        """
        Compute SVD of the user ratings matrix.
        """
        U, s, Vt = sparsesvd.sparsesvd(urm, K)
        dim = (len(s), len(s))
        S = numpy.zeros(dim, dtype=numpy.float32)
        for i in range(0, len(s)):
            S[i,i] = math.sqrt(s[i])
        U = scipy.sparse.csc_matrix(numpy.transpose(U), dtype=numpy.float32)
        S = scipy.sparse.csc_matrix(S, dtype=numpy.float32)
        Vt = scipy.sparse.csc_matrix(Vt, dtype=numpy.float32)
        return U, S, Vt

    def plot_precision_recall(
        self,
        m1_precision_list,
        m1_recall_list,
        m1_label,
        m2_precision_list,
        m2_recall_list,
        m2_label):
        """
        Method to generate precision and recall curve.
        """
        pylab.clf()    
        pylab.plot(m1_recall_list, m1_precision_list, label=m1_label)
        pylab.plot(m2_recall_list, m2_precision_list, label=m2_label)
        pylab.xlabel('Recall')
        pylab.ylabel('Precision')
        pylab.ylim([0.0, 0.20])
        pylab.xlim([0.0, 0.20])
        pylab.title('Precision-Recall curve')
        pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
        pylab.show()
