from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data_pdfs/test_wipro.pdf")
pages = loader.load_and_split()

print(pages[0])