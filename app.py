import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are a Stock Annual Report Question and Answer bot.
You will be provided context to a company's annual report.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# mistralai/Mistral-7B-Instruct-v0.1/2
# mistralai/Mixtral-8x7B-Instruct-v0.1
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",max_new_tokens=1000,temperature=0.7,top_k=50,top_p=0.95)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Give me information on business ratios/information"))