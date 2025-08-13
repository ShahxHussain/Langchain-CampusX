from langchain_community.document_loaders import PyPDFLoader
import os


file_path = os.path.join(os.path.dirname(__file__), 'intro.pdf')
loader = PyPDFLoader(file_path)
docs = loader.load()
print(docs)
print(len(docs)) # Number of pages

print(len(docs))  # Print the number of documents loaded
print(docs[0])  # Print the first document in the list
print(type(docs[0]))  # Print the type of the first document