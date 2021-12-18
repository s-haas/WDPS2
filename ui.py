from rottentomatoes import search_for_movies, reviews_for_movie
from ui_chooser import choose


def search_for_movie():
    movie_query = input("Search for a movie:\n")
    return search_for_movies(movie_query)

def choose_movie(movie_suggestions):
    return choose(movie_suggestions, lambda movie: f"{movie['title']}")

def choose_review(movie):
    reviews = reviews_for_movie(movie['id'])
    return choose(reviews, lambda rv: f"{rv['critic_name']}: {rv['content'][:100]}", prompt=f"\nReviews for {movie['title']}")



movies = search_for_movie()
chosen_movie = choose_movie(movies)
review = choose_review(chosen_movie)

print(f"\n{review['critic_name']} is typcally very negative (-0.968)") # MOCK