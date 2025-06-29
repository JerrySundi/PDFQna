from transformers import AutoModel
from dotenv import load_dotenv
import chromadb
import numpy as np
from logger import logger
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

def create_embeddings(arr):
    logger.info(f"Creating embeddings for {len(arr)} items")
    embedding = model.encode(arr)
    return embedding

# def encode(arr):
#     embeddings = model.encode(arr)
#     return embeddings

# def create_embeddings(arr):
#     logger.info(f"Creating embeddings for {len(arr)} items using multithreading")
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         embeddings = executor.map(encode, arr)
#     return embeddings

def store_embeddings(collection_name, texts, embeddings, metadata=None):
    logger.info(f"Storing embeddings in collection: {collection_name}")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    ids = [str(i) for i in range(len(texts))]

    collection.upsert(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadata,
        ids=ids
    )
    logger.info(f"Stored {len(texts)} embeddings")

def similarity_search(collection_name, query, k=4):
    logger.info(f"Performing similarity search in collection: {collection_name}")
    collection = chroma_client.get_collection(collection_name)

    query_embedding = create_embeddings(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    similar_items = []

    for i in range(k):
        similar_items.append({
            'document': results['documents'][0][i],
            'score': 1- results['distances'][0][i]
        })

    logger.info(f"Found {len(similar_items)} similar items")
    return similar_items