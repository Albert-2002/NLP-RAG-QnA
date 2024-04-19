from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


loader = PyMuPDFLoader("data_pdfs/idfc_fy21.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=501000, chunk_overlap=1000)

all_splits = text_splitter.split_documents(data)

embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./chroma_db")