import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",max_new_tokens=1500,
                                             temperature=0.5,top_k=80,top_p=0.95,streaming=True)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("-----------------------------------------------------------------------------")

if os.path.exists('./chroma_db') == True:
    print("DB Exists - Accessing......")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    print("DB Does not Exist - Creating......")
    loader = PyMuPDFLoader("data_pdfs/idfc_fy21.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=501000, chunk_overlap=20000)
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./chroma_db")

print("-----------------------------------------------------------------------------")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print("Welcome to the PDF Chatbot!")
print("-----------------------------------------------------------------------------")

i = 0
x1 = "Null"
while i == 0:
    question = str(input("[1] Enter a question: \n[2] Enter 'Exit' to quit: \n[3] Enter 'Context' to view last output's context: \n\n"))
    if question == 'Exit':
        print("-x-x-x-x-x-x-x-x-x-x-x-")
        print("Thank you!")
        print("-x-x-x-x-x-x-x-x-x-x-x-")
        break
    elif question == 'Context':
        if x1 == "Null":
            print("No context available!")
            print("-----------------------------------------------------------------------------")
        else:
            for document in x1["context"]:
                print('PDF Page Numbers: ',document.metadata.get('page'))
                print("-----------------------------------------------------------------------------")
    else:
        x1 = conversational_rag_chain.invoke({"input": question},
                                             config={"configurable": {"session_id": "abc123"}},)
        print(x1["answer"])
        print("-----------------------------------------------------------------------------")