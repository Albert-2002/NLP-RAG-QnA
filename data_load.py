import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-v0.1")

loader = PyMuPDFLoader("data_pdfs/idfc_fy21.pdf")
data = loader.load()

# print(data[111].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

all_splits = text_splitter.split_documents(data)

# print(all_splits[0])

# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=HF_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
# )

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

print(example_messages[0].content)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is this pdf about?"):
    print(chunk, end="", flush=True)