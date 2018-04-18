import pandas as pd
import json
import operator

all_movie_data = pd.read_csv('tmdb_5000_movies.csv')
all_credit_data = pd.read_csv('tmdb_5000_credits.csv')

zero_revenue_indices = all_movie_data.revenue != 0
all_movie_data = all_movie_data[zero_revenue_indices]
all_credit_data = all_credit_data[zero_revenue_indices]

production_companies = all_movie_data['production_companies'].value_counts().keys()

prod_comps = {}
prod_comps_names = {}
for prod_comp_json in production_companies:
    prod_comp_list = json.loads(prod_comp_json)
    for prod_comp in prod_comp_list:
        if prod_comp['id'] not in prod_comps:
            prod_comps[prod_comp['id']] = 0
            prod_comps_names[prod_comp['id']] = prod_comp['name']

        prod_comps[prod_comp['id']] += 1

sorted_comps = sorted(prod_comps.items(), key=operator.itemgetter(1), reverse=True)

top_companies = {}
for id, freq in sorted_comps[:29]:
    top_companies[str(id)] = prod_comps_names[id]
    

print(str(list(top_companies.values())))
print('done')

