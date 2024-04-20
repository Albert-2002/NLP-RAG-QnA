from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

import os
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

class Agent:
    def __init__(self, huggingfacehub_api_token: str | None = None) -> None:
        # if HUGGINGFACEHUB_API_TOKEN is None, then it will look the enviroment variable HUGGINGFACEHUB_API_TOKEN
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=501000, chunk_overlap=20000)
        # mistralai/Mistral-7B-Instruct-v0.1/2
        # mistralai/Mixtral-8x7B-Instruct-v0.1

        self.llm = llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",max_new_tokens=1500,
                                             temperature=0.5,top_k=80,top_p=0.95,streaming=True)

        self.chat_history = None
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            response = self.chain({"question": question, "chat_history": self.chat_history})
            response = response["answer"].strip()
            self.chat_history.append((question, response))
        return response

    def ingest(self, file_path: os.PathLike) -> None:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)

        if self.db is None:
            self.db = Chroma.from_documents(splitted_documents, self.embeddings)
            self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever(search_type="similarity", 
                                                                                              search_kwargs={"k": 16}))
            self.chat_history = []
        else:
            self.db.add_documents(splitted_documents)

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history = None