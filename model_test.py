import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# print(HF_TOKEN)

########################################################################################################

template = """Question: {question}

Answer: Let's see here."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-v0.1", max_new_tokens=64)

# you can use  Encoder-Decoder Model ("text-generation") or  Encoder-Decoder Model ("text2text-generation")
llm_chain = LLMChain(prompt=prompt,llm=llm)

print("---"*20)
question = str(input("Enter the question: \n"))
print("---"*20)

print(llm_chain.invoke(question).get('text'))

########################################################################################################