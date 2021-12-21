# Analyzing sentiments of movie reviewers

*by Stefan Haas, Leonhard Wattenbach and Willem van der Spek*

## Abstract

Movies reviews help to find out if a certain movie will probably be good or not. We built a tool that analyzes the sentiments of all reviews per reviewer by using machine learning methods. The user of this tool can then after reeding a review see if this reviewer is writing in general rather positive or rather negative reviews.

## Introduction

There are several platforms for getting information about movies. [Rotten Tomatoes](https://rottentomatoes.com). The platform shows what the most popular movies are at the moment and also offers the possibility to search for a specific movie. For every movie you can see general information like cast, release date and runtime but also snippets from the movie and reviews from professional critics.

In this project, we focus on these critic reviews. Using machine learning techniques, we classify every review of a selected reviewer as either positive or negative. This classification has a certainty. Based on the classifications, we calculate an overall positivity score for the reviewer that lies between -1 (very negative) and 1 (very positive). This score helps the user to assess the relevance of a certain review. For instance if a movie has many negative reviews and a positive review that is written by a reviewer with a high positivity score, then the user can assume that the movie will probably not be the best movie.

## Training the model
TODO

## Data
### Training data
TODO

### Runtime data
The data that is used during runtime comes directly from the [Rotten Tomatoes site](https://rottentomatoes.com). We load and parse the webpages using [BeautifulSoup](https://pypi.org/project/beautifulsoup4/). The file `rottentomatoes.py` is a Python-wrapper for three relevant endpoints:

- `search_for_movies(query)`: Returns a list of movie suggestions for a given search query.
- `reviews_for_movie(movie_id)`: Returns a list of critic reviews of a certain movie. The passed movie id can be obtained using the `id` attribute of a movie search suggestion.
- `reviews_of_critic(critic_id)`: Returns a list of reviews for a certain critic. The critic id can be obtained using the `critic_id` attribute of a review from a reviews for movie request.

## Running the software
The software can be run using the command:

`python3 ui.py`

You can then enter a movie title, select the movie from a list and select a review of this movie from a list.