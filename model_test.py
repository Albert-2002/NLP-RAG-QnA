import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# print(HF_TOKEN)

""" llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)
print("---"*20)
prmpt = str(input("Enter the prompt: \n"))
print("---"*20)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content=prmpt
    ),
]

chat_model = ChatHuggingFace(llm=llm)

res = chat_model.invoke(messages)
print(res.content) """

########################################################################################################

template = """Question: {question}

Answer: Let's see here."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = HuggingFaceEndpoint(repo_id="databricks/dolly-v2-3b", temperature=0.5, max_length=64)

# you can use  Encoder-Decoder Model ("text-generation") or  Encoder-Decoder Model ("text2text-generation")
llm_chain = LLMChain(prompt=prompt,llm=llm)

print("---"*20)
question = str(input("Enter the question: \n"))

print(llm_chain.invoke(question).get('text'))
print("---"*20)

########################################################################################################