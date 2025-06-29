import os 
from openai import AzureOpenAI
from embeddings import similarity_search
import requests
import wikipedia
from serpapi import GoogleSearch
from logger import logger

azure_endpoint = "https://ai-proxy.lab.epam.com"
api_key = os.getenv("OPENAI_API_KEY")
api_version = "2023-07-01-preview"
deployment_name = "gpt-35-turbo"

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

class WebAgent:
    def __init__(self):
        self.session = requests.Session()

    def search_wikipedia(self, query):
        logger.info(f"Searching Wikipedia for: {query}")
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                logger.info("No results found on Wikipedia")
                return "No results found on Wikipedia."
            page = wikipedia.page(search_results[0], auto_suggest=False)
            logger.info(f"Found Wikipedia page: {page.title}")
            return f"Summary: {page.summary[:500]}...\nURL: {page.url}"
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Wikipedia disambiguation error for query: {query}")
            return f"Multiple matches found. Please be more specific. Options: {', '.join(e.options[:5])}"
        except Exception as e:
            logger.error(f"Error in Wikipedia search: {str(e)}")
            return f"An error occurred: {str(e)}"

    def search_and_extract(self, query):
        logger.info(f"Performing web search for: {query}")
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "organic_results" in results and results["organic_results"]:
                top_result = results["organic_results"][0]
                title = top_result.get("title", "No title found")
                snippet = top_result.get("snippet", "No snippet found")
                link = top_result.get("link", "No URL found")
                logger.info(f"Found web search result: {title}")
                return f"Title: {title}\nSummary: {snippet}\nURL: {link}"
            else:
                logger.info("No search results found")
                return "No search results found."
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return f"An error occurred during web search: {str(e)}"

def get_llm_response(messages):
    logger.info("Getting LLM response")
    answer_response = client.chat.completions.create(
        model=deployment_name,
        messages=messages
    )
    return answer_response.choices[0].message.content

def get_external_info(query):
    logger.info(f"Getting external info for query: {query}")
    agent = WebAgent()
    wikipedia_result = agent.search_wikipedia(query)
    web_result = agent.search_and_extract(query)
    if wikipedia_result == "No results found on Wikipedia.":
        return f"Web Search:\n{web_result}"
    return f"Wikipedia:\n{wikipedia_result}\n\nWeb Search:\n{web_result}"

def get_answer(question):
    logger.info(f"Processing question: {question}")
    raw_context = similarity_search('test', question)
    len_contexts = len(raw_context)
    mx_score = 0
    context = ""

    for i in range(len_contexts):
        if(raw_context[i]['score']<1 and raw_context[i]['score']>= 0):
            context += raw_context[i]['document']
            mx_score = max(mx_score,raw_context[i]['score'])
    
    with open('answer.txt','w') as file:
        file.write(str(mx_score))

    if len(context) > 0:
        cleaned_context = context.replace('\n', ' ')

        messages = [
            {"role":"system","content":"You are a helpful assistant that answers the question based on the provided content. If the question is related but not directly answered by the content, say so and provide an explanation based on your general knowledge."},
            {"role":"user","content":f"question: {question} , paragraph: {cleaned_context}"}
        ]

        final_answer = get_llm_response(messages)

        if mx_score < .38:
            logger.info("Low confidence score, fetching external info")
            external_info = get_external_info(question)
            final_answer = '\n\n'.join([final_answer, external_info])

        logger.info("Answer generated successfully")
        return final_answer
    else:
        logger.warning("Question not related to the context of the documents")
        return "The question is no way related to the context of the document/s, kindly ask questions related/relevant to the docs"