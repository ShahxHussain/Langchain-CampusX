import os  
from langchain_community.document_loaders import TextLoader 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash'
)

template = PromptTemplate(
    template= "Generate a summary of the following poem \n {poem}"
)

parser = StrOutputParser()

# Build the absolute path to 'poem.txt' relative to this script's location
file_path = os.path.join(os.path.dirname(__file__), 'poem.txt')
loader = TextLoader(file_path, encoding='utf-8')  # Initialize the TextLoader with the file path and encoding

# Load the contents of the file as documents
docs = loader.load()
print(docs)  # Print the list of loaded documents
print(type(docs))  # Print the type of the 'docs' object

print(len(docs))  # Print the number of documents loaded
print(docs[0])  # Print the first document in the list
print(type(docs[0]))  # Print the type of the first document

chain = template | model | parser
print(chain.invoke({'poem': docs[0].page_content})) # Summarize the poem.txt