import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader= TextLoader("C:/Users/prima/langchain/langchain-course/mediumblog1.txt",
                       encoding='UTF-8')
    document= loader.load()

    print("splitting ...")
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts= text_splitter.split_documents(document)
    print(f"created{len(texts)} chunks")
    print(os.getenv(""))
    
    embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")

