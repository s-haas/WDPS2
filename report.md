# Analyzing sentiments of movie reviewers

*by Stefan Haas, Leonhard Wattenbach and Willem van der Spek*

***Abstract.**
Movie reviews can help to find out if a certain movie is worth watching or not. We built a prototype for a tool that analyzes the sentiments of all reviews per reviewer by using machine learning methods. The user of this tool can then after reading a review see if this reviewer usually writes rather positive or rather negative reviews.*

## Introduction

There are several platforms for getting information about movies. One of them is [Rotten Tomatoes](https://rottentomatoes.com). The platform shows what the most popular movies are at the moment and also offers the possibility to search for a specific movie. For every movie you can see general information like cast, release date and runtime but also snippets from the movie and reviews from professional critics.

In this project, we focus on these critic reviews. Using machine learning techniques, we classify every review of a selected reviewer as either positive or negative. This classification has a certainty. Based on the classifications, we calculate an overall positivity score for the reviewer that lies between -1 (very negative) and +1 (very positive). This score helps the user to assess the relevance of a certain review. For instance if a movie has many negative reviews and the only positive review is written by a reviewer with a high positivity score, then the user can assume that the movie will probably not be the best movie.

## Idea
**TODO:**
- Explain which ML techniques we used to build the classifier/regressor (and why)
- Explain which data we used to train the models and how we got this data (if I understood it correctly via BERT as gold standard to generate labeled data)

## Movie data
The data that is used during runtime comes directly from the [Rotten Tomatoes site](https://rottentomatoes.com). We load and parse the webpages using [BeautifulSoup](https://pypi.org/project/beautifulsoup4/). The file `rottentomatoes.py` is a Python-wrapper and has three endpoints:

- `search_for_movies(query)`: Returns a list of movie suggestions for a given search query. The movies have an id (`id`) and some general attributes (`title`, `link`, `year` and `cast`).
- `reviews_for_movie(movie_id)`: Returns a list of critic reviews of a certain movie. The passed movie id can be obtained using the `id` attribute of a movie search suggestion. The reviews contain id and name of the reviewer (`critic_id` and `critic_name`) as well as the content of the review snippet (`content`).
- `reviews_of_critic(critic_id)`: Returns a list of reviews for a certain critic. The critic id can be obtained using the `critic_id` attribute of a review from a reviews for movie request. The reviews contain id and name of the reviewed movie (`movie_id` and `movie_name`) and the content of the review snippet (`content`).

## How to run the software
The software can be run using the command:

    python3 ui.py

You can then enter a movie title, select the movie from a list and select a review of this movie from a list. The positivity score of the reviewer of the selected review is then presented as a number between -1 (very negative) and +1 (very positive).