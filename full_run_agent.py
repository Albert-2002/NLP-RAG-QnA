from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_groq import ChatGroq

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

import os
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
GROQ_TOKEN = os.environ.get("GROQ_API_KEY")

# groq_llm = ChatGroq(temperature=0, groq_api_key=GROQ_TOKEN, model_name="llama3-8b-8192")

def clean_string(text):
    # Remove all occurrences of "AI:" and "</s>"
    cleaned_text = text.replace("AI:", "").replace("</s>", "").replace("Assistant:","").replace("Bot:","")
    # Optionally, trim leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

### Answer question ###
qa_system_prompt = """You are an Stock Annual Report Question and Answer bot.
Answer with respect to the company's name.
Use the following pieces of context to answer the question at the end.
Perform calculations when needed and asked in the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.
Always say "Thanks for asking!" at the end of the answer.

{context}"""

# AGENT

class Agent:
    def __init__(self, huggingfacehub_api_token: str | None = None) -> None:
        # if HUGGINGFACEHUB_API_TOKEN is None, then it will look the enviroment variable HUGGINGFACEHUB_API_TOKEN
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=501000, chunk_overlap=20000)
        # mistralai/Mistral-7B-Instruct-v0.1/2
        # mistralai/Mixtral-8x7B-Instruct-v0.1

        self.llm = llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",max_new_tokens=1500,
                                            temperature=0.5,top_k=80,top_p=0.95,streaming=True)
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            if question.endswith('[CONTEXT]'):
                question = question.replace('[CONTEXT]','')
                response = self.chain.invoke({"input": question}, config={"configurable": {"session_id": "abc123"}},)
                context_response = ""
                for document_p in response["context"]:
                    context_response += str('PDF Page Number: '+str(document_p.metadata.get('page'))+'\n')
                response = clean_string(response["answer"])+'\nCONTEXT OBTAINED FROM:\n'+context_response
            else:
                response = self.chain.invoke({"input": question}, config={"configurable": {"session_id": "abc123"}},)
                response = clean_string(response["answer"])

        return response

    def ingest(self, file_path: os.PathLike) -> None:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)

        if self.db is None:
            self.db = Chroma.from_documents(splitted_documents, self.embeddings)

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),])
            # retrieev = self.db.as_retriever()
            secondary_retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 16})
            history_aware_retriever = create_history_aware_retriever(self.llm,secondary_retriever,contextualize_q_prompt)
            # history_aware_retriever = create_history_aware_retriever(
            # self.llm, MultiQueryRetriever.from_llm(retriever=retrieev,llm=groq_llm),contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages(
            [("system", qa_system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])

            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            ### Statefully manage chat history ###
            store = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id] = ChatMessageHistory()
                return store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="input",
                                                                history_messages_key="chat_history",output_messages_key="answer",)

            self.chain = conversational_rag_chain
        else:
            self.db.add_documents(splitted_documents)

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history = None

# Set logging for the queries
# import logging

# logging.basicConfig()
# print(logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO))