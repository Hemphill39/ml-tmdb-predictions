import pandas as pd
import json

all_movie_data = pd.read_csv('tmdb_5000_movies.csv')
all_credit_data = pd.read_csv('tmdb_5000_credits.csv')

zero_revenue_indices = all_movie_data.revenue == 0
all_movie_data = all_movie_data[zero_revenue_indices]
all_credit_data = all_credit_data[zero_revenue_indices]


