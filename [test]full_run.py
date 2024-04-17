import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain import hub
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# loader = PyMuPDFLoader("data_pdfs/idfc_fy21.pdf")
# data = loader.load()

# # print(data[111].page_content)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# all_splits = text_splitter.split_documents(data)

# # print(all_splits[0])

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./chroma_db")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are an Stock Annual Report Question and Answer bot.
Answer with respect to the company's name.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1",max_new_tokens=64,temperature=0.7,top_k=50,top_p=0.95)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# for chunk in rag_chain.stream("what is the pdf about?"):
#     print(chunk, end="", flush=True)

print(rag_chain.invoke("What is the overall revenue of the bank?"))