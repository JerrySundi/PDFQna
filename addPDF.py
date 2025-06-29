from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import create_embeddings, store_embeddings
from logger import logger

def process_pdfs(pdf_directory):
    logger.info(f"Processing PDFs from directory: {pdf_directory}")
    # Load environment variables
    load_dotenv()

    # Read docs
    file_loader = PyPDFDirectoryLoader(pdf_directory)
    docs = file_loader.load()
    logger.info(f"Loaded {len(docs)} document/s")

    # Divide the docs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split documents into {len(chunks)} chunks")

    texts = [chunk.page_content for chunk in chunks]
    logger.info("Creating embeddings")
    embeddings = create_embeddings(texts)

    # Storing embeddings
    logger.info("Storing embeddings")
    store_embeddings("test", texts, embeddings)
    logger.info("Finished processing PDFs")