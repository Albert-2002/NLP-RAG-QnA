import streamlit as st
import os
import tempfile
from dotenv import load_dotenv, find_dotenv
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
from langchain.vectorstores import Chroma

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

def create_chroma_vectorstore(pdf_path):
    db_path = f"./{os.path.splitext(os.path.basename(pdf_path))[0]}_chroma_db"
    if os.path.exists(db_path):
        embeddings = None
        VectorStore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = PyMuPDFLoader(pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=501000, chunk_overlap=1000)
        all_splits = text_splitter.split_documents(data)
        VectorStore = Chroma.from_documents(all_splits, embedding=embeddings)
    return VectorStore

st.header("Chat with me!") 
pdf = st.file_uploader("Upload your pdf", type='pdf')

if pdf is not None:
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=1500,
                                         temperature=0.5, top_k=80, top_p=0.95, streaming=True)
    
    # Save uploaded PDF to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf.read())
        pdf_path = temp_file.name
    
    VectorStore = create_chroma_vectorstore(pdf_path)

    retriever = VectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

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

    st.write("Welcome to the PDF Chatbot!")
    st.write("-----------------------------------------------------------------------------")

    i = 0
    x1 = "Null"
    question_input_id = "question_input"

    while True:
        question = st.text_input("[1] Enter a question: \n[2] Enter 'Exit' to quit: \n[3] Enter 'Context' to view last output's context: \n\n", key=question_input_id)

        if question == 'Exit':
            st.write("Thank you!")
            break
        elif question == 'Context':
            if 'x1' not in locals():
                st.write("No context available!")
            else:
                for document in x1["context"]:
                    st.write('PDF Page Numbers: ', document.metadata.get('page'))
        else:
            x1 = conversational_rag_chain.invoke({"input": question}, config={"configurable": {"session_id": "abc123"}})
            st.write(x1["answer"])

        st.write("-----------------------------------------------------------------------------")
