import requests
import json
from bs4 import BeautifulSoup


def get_parsed_website(url):
    return BeautifulSoup(requests.get(url).text, features="lxml")

def reviews_for_movie(movie_id):
    doc = get_parsed_website(f"https://www.rottentomatoes.com/m/{movie_id}/reviews")
    rows = doc.find_all(True, ["review_table_row"])

    def extract_information(row):
        critic_element = row.find(True, {"data-qa": "review-critic-link"})
        return {
            "critic_id": critic_element["href"][len("/critics/"):],
            "critic_name": critic_element.text.strip(),
            "content": row.find(True, "the_review").text.strip()
        }

    return [extract_information(row) for row in rows]

def reviews_of_critic(critic_id):
    doc = get_parsed_website(f"https://www.rottentomatoes.com/critics/{critic_id}/movies")
    graph = json.loads(doc.find("script", {"type": "application/ld+json"}).text)["@graph"]
    reviews = [row["item"] for row in graph[1]["itemListElement"]]
    def extract_information(review):
        return {
            "content": review["reviewBody"],
            "movie_id": review["itemReviewed"]["sameAs"][len("/m/"):],
            "movie_name": review["itemReviewed"]["name"]
        }

    return [extract_information(review) for review in reviews]

print(json.dumps(reviews_of_critic("cate-young")[:5], indent=2))