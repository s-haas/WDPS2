import numpy as np
import json
import tempfile

from rottentomatoes import search_for_movies, reviews_for_movie, reviews_of_critic
from ui_chooser import choose

def json_to_file(json_obj, fpath="data/critic_reviews.json"):
    if not fpath.endswith(".json"):
        return False
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)
    return True


def gather_result(pred):
    '''
    Gather the sentiment according to the model's predictions. 
    '''
    sents = ["Very Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Very Positive"]
    mean_sentiment = np.mean(pred)
    sent_text = sents[(np.abs(np.linspace(0, 1, len(sents)) 
                              - mean_sentiment)).argmin()] # Gather sentiment string closest to mean.
    return mean_sentiment, sent_text


def main():
    movies = search_for_movie()
    chosen_movie, reviews = choose_movie(movies)
    review = choose_review(chosen_movie, reviews)

    # print(f"\n{review['critic_name']} is typically very negative (-0.968)") # MOCK
    json_to_file(reviews_of_critic(review["critic_id"]))
    pred = model()
    sentiment, sentiment_text = gather_result(pred)
    print(f"\n{review['critic_name']} is typically {sentiment_text} ({sentiment})")

def search_for_movie():
    while(True):
        movie_query = input("Search for a movie:\n")
        movies = search_for_movies(movie_query)

        if len(movies) == 0:
            print(f"No movies found for query “{movie_query}”. Please try a different query.")
        else:
            return movies


def choose_movie(movie_suggestions):
    while(True):
        movie = choose(movie_suggestions, lambda movie: f"{movie['title']} ({movie['year']}; {', '.join(movie['cast'])})")
        reviews = reviews_for_movie(movie['id'])

        if len(reviews) == 0:
            print(f"{movie['title']} has no reviews. Please choose a different movie.")
        else:
            return (movie, reviews)
    

def choose_review(movie, reviews):
    return choose(reviews, lambda rv: f"{rv['critic_name']}: {rv['content']}", prompt=f"\nReviews for {movie['title']}")    


if __name__ == "__main__":
    main()