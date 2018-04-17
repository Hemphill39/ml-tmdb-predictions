library(readr)
library(ggplot2)
tmdb_5000_movies <- read_csv("~/Projects/ml-tmdb-predictions/tmdb_5000_movies.csv")

# make some graphs to look for correlations
qplot(x=log(popularity), y=log(revenue), data=tmdb_5000_movies)

