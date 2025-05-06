from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")

    # Get document and load the content of it
    loader = TextLoader(
        "/home/abdhulla-k/aside/loading/ice_breaker/rag-intro/mediumblog1.txt"
    )
    document = loader.load()

    # Split the content
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    PineconeVectorStore.from_documents(
        documents=texts[:5], embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
