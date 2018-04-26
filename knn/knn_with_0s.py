import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

NUM_OF_CLASSES = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
K_VALUES = [2, 3, 5, 7, 9]

COLUMN_NAMES = [
    'budget',
    'popularity',
    'runtime',
    'vote_average',
    'vote_count'
]

COLUMNS_TO_KEEP = COLUMN_NAMES

COLUMNS_TO_POP = [
    'genres',
    'homepage',
    'keywords',
    'id',
    'keywords',
    'original_language',
    'original_title',
    'overview',
    'production_countries',
    'production_companies',
    'release_date',
    'spoken_languages',
    'status',
    'tagline',
    'title'
]

all_movie_data = 0

def load_data():
    global all_movie_data 
    all_movie_data = pd.read_csv('../tmdb_5000_movies.csv', na_values=[])

def filter_data(classes):
    global all_movie_data
    load_data()
    for key in all_movie_data.columns:
        if key not in COLUMNS_TO_KEEP and key != 'revenue':
            all_movie_data.pop(key)

    all_movie_data = all_movie_data[~(all_movie_data.isnull().any(axis=1))]
    # all_credit_data = pd.read_csv('tmdb_5000_credits.csv')

    # bin the revenue data
    all_movie_data['revenue'] = pd.cut(all_movie_data['revenue'], len(range(classes)), labels=range(classes)).cat.codes.astype(np.int64)
    train_data = all_movie_data.sample(int(0.8 * len(all_movie_data)))
    train_data_labels = train_data.pop('revenue')
    test_data = all_movie_data.drop(train_data.index)
    test_data_labels = test_data.pop('revenue')
    return (train_data, train_data_labels), (test_data, test_data_labels)

def get_accuracy(knn_result_indices, train_data, train_data_labels, test_data_labels):
    dimensions = np.shape(knn_result_indices)
    train_label_list = train_data_labels.tolist()
    for i in range(dimensions[0]):
        knn_result = []
        for j in range(dimensions[1]):
            knn_result.append(train_label_list[knn_result_indices[i][j]])
        print(str(knn_result) + str(test_data_labels.tolist()[i]))

def main():
    load_data()
    global all_movie_data
    k_samples = 5
    accuracies = []
    for classes in NUM_OF_CLASSES:
        (train, train_labels), (test, test_labels) = filter_data(classes)
        class_accuracies = []
        for k in K_VALUES:
            k_accuracies = []
            for i in range(k_samples):
                knn = KNeighborsClassifier(n_neighbors = k, weights='uniform', algorithm='auto')
                knn.fit(train, train_labels)
                test_predict = knn.predict(test)
                k_accuracies.append(accuracy_score(test_labels, test_predict))
            avg_accuracy = sum(k_accuracies) / k_samples
            class_accuracies.append(avg_accuracy)
            print('Accuracy for K=' + str(k) + ' Num Classes = ' + str(classes) + ' = ' + str(avg_accuracy))
        print

main()	
