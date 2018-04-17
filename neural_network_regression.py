import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import json
import tensorflow as tf


CLASSIFICATIONS = [
    'LOW',
    'LOW-MED',
    'MED',
    'MED-HIGH',
    'HIGH',
    'VERY-HIGH',
    'ULTRA-HIGH'
]

CLASSIFICATIONS_FLOAT = [
    0,
    1,
    2,
    3,
    4,
    5,
    6
]

COLUMN_NAMES = [
    'budget',
    'popularity',
    'runtime',
    'vote_average',
    'vote_count'
]

COLUMNS_TO_POP = [
    'genres',
    'homepage',
    'keywords',
    'id',
    'keywords',
    'original_language',
    'original_title',
    'overview',
    'production_companies',
    'production_countries',
    'release_date',
    'spoken_languages',
    'status',
    'tagline',
    'title'
]

# movies = pd.read_csv('tmdb_5000_movies.csv')

# credits = pd.read_csv('tmdb_5000_credits.csv')

# as a test lets grab budget, popularity, runtime, vote_average, and vote_count
# labels will be revenue
def load_data():

    all_movie_data = pd.read_csv('tmdb_5000_movies.csv', na_values=[])
    for key in all_movie_data.columns:
        if key not in COLUMN_NAMES and key != 'revenue':
            all_movie_data.pop(key)

    # TODO remove any row with null data
    all_movie_data = all_movie_data[~(all_movie_data.isnull().any(axis=1))]
    # all_credit_data = pd.read_csv('tmdb_5000_credits.csv')

    zero_revenue_indices = all_movie_data.revenue != 0
    all_movie_data = all_movie_data[zero_revenue_indices]
    # all_credit_data = all_credit_data[zero_revenue_indices]

    # bin the revenue data
    all_movie_data['revenue'] = pd.cut(all_movie_data['revenue'], len(CLASSIFICATIONS_FLOAT), labels=CLASSIFICATIONS_FLOAT).cat.codes.astype(np.int64)

    train_data = all_movie_data.sample(int(0.8 * len(all_movie_data)))
    train_data_labels = train_data.pop('revenue')
    test_data = all_movie_data.drop(train_data.index)
    test_data_labels = test_data.pop('revenue')

    return (train_data, train_data_labels), (test_data, test_data_labels)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (dict(features), labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def main():
    # load in the training and test data
    (train_data, train_labels), (test_data, test_labels) = load_data()

    feature_columns = []
    for key in COLUMN_NAMES:
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    hidden_units = [10, 10]

    classifier = tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        n_classes=len(CLASSIFICATIONS_FLOAT),
        hidden_units = hidden_units
    )

    batch_size = 10
    train_steps = 1000

    classifier.train(input_fn=lambda:train_input_fn(train_data, train_labels, batch_size), steps=train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_data, test_labels, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


main()