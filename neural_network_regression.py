import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import json
import tensorflow as tf

PRODUCTION_COMPANIES = ['Warner Bros.', 'Universal Pictures', 'Paramount Pictures', 'Twentieth Century Fox Film Corporation', 'Columbia Pictures', 'New Line Cinema', 'Relativity Media','Touchstone Pictures', 'Columbia Pictures Corporation', 'Village Roadshow Pictures', 'Metro-Goldwyn-Mayer (MGM)', 'Regency Enterprises', 'Walt Disney Pictures', 'Miramax Films', 'DreamWorks SKG', 'Dune Entertainment', 'Canal+', 'Fox Searchlight Pictures', 'United Artists', 'Summit Entertainment', 'Lionsgate', 'Working TitleFilms', 'TriStar Pictures', 'Fox 2000 Pictures', 'Amblin Entertainment', 'StudioCanal', 'New Regency Pictures', 'The Weinstein Company', 'Legendary Pictures']

CLASSIFICATIONS = [
    'LOW',
    'LOW-MED',
    'MED',
    'MED-HIGH',
    'HIGH',
    'VERY-HIGH',
    'ULTRA-HIGH'
]

CLASSES = 20
CLASSIFICATIONS_FLOAT = range(CLASSES)

COLUMN_NAMES = [
    'budget',
    'popularity',
    'runtime',
    'vote_average',
    'vote_count'
]

COLUMNS_TO_KEEP = COLUMN_NAMES + [
    'production_companies'
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
        if key not in COLUMNS_TO_KEEP and key != 'revenue':
            all_movie_data.pop(key)

    all_movie_data = all_movie_data[~(all_movie_data.isnull().any(axis=1))]
    # all_credit_data = pd.read_csv('tmdb_5000_credits.csv')

    zero_revenue_indices = all_movie_data.revenue != 0
    all_movie_data = all_movie_data[zero_revenue_indices]
    # all_credit_data = all_credit_data[zero_revenue_indices]

    for idx, prod_comp in enumerate(PRODUCTION_COMPANIES):
        all_movie_data['prod_comp' + str(idx)] = all_movie_data.apply(lambda row: 1 if prod_comp in row.production_companies else 0, axis=1)

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

    # for idx, prod_comp in enumerate(PRODUCTION_COMPANIES):
    #     feature_columns.append(tf.feature_column.numeric_column(key='prod_comp' + str(idx)))

    hidden_units = [64, 32]

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