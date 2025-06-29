from serpapi import GoogleSearch
import os 
from logger import logger

serpapi_key = os.getenv("SERPAPI_API_KEY")

def func(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    for result in results:
        print(result,'\n\n')

func("tell me about tensorflow")
