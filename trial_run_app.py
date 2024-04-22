import os
import tempfile
import streamlit as st
from streamlit_chat import message
from trial_run_agent import Agent

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["agent"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["agent"].forget()  # to reset the knowledge base
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["agent"].ingest(file_path)
        os.remove(file_path)


def is_HUGGINGFACEHUB_API_TOKEN_set() -> bool:
    return len(st.session_state["HUGGINGFACEHUB_API_TOKEN"]) > 0


def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        if is_HUGGINGFACEHUB_API_TOKEN_set():
            st.session_state["agent"] = Agent(st.session_state["HUGGINGFACEHUB_API_TOKEN"])
        else:
            st.session_state["agent"] = None

    st.header("ChatPDF")

    if st.text_input("HuggingFace API Key", value=st.session_state["HUGGINGFACEHUB_API_TOKEN"], key="input_HUGGINGFACEHUB_API_TOKEN", type="password"):
        if (
            len(st.session_state["input_HUGGINGFACEHUB_API_TOKEN"]) > 0
            and st.session_state["input_HUGGINGFACEHUB_API_TOKEN"] != st.session_state["HUGGINGFACEHUB_API_TOKEN"]
        ):
            st.session_state["HUGGINGFACEHUB_API_TOKEN"] = st.session_state["input_HUGGINGFACEHUB_API_TOKEN"]
            if st.session_state["agent"] is not None:
                st.warning("Please, upload the files again.")
            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            st.session_state["agent"] = Agent(st.session_state["HUGGINGFACEHUB_API_TOKEN"])

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_HUGGINGFACEHUB_API_TOKEN_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", disabled=not is_HUGGINGFACEHUB_API_TOKEN_set(), on_change=process_input)

    st.divider()

if __name__ == "__main__":
    main()