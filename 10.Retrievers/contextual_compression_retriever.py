from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

# Sample long documents with relevant + irrelevant sentences
docs = [
    Document(page_content="Regular exercise helps improve cardiovascular health and strengthens muscles. "
                          "Many people also like watching cat videos on YouTube during their free time. "
                          "Daily workouts reduce the risk of obesity and boost energy levels."),
    
    Document(page_content="A balanced diet with proteins, carbs, and healthy fats supports overall body health. "
                          "I once traveled to Paris and enjoyed the Eiffel Tower views. "
                          "Consuming fruits and vegetables provides essential vitamins and minerals."),
    
    Document(page_content="Drinking enough water daily is essential for hydration and maintaining energy levels. "
                          "Some people also prefer soda, but it is not as healthy. "
                          "Water helps regulate body temperature and supports digestion."),
    
    Document(page_content="Managing stress through meditation, yoga, and deep breathing enhances both mental "
                          "and physical health. My favorite hobby is playing video games at night. "
                          "Stress reduction techniques can improve sleep quality and lower blood pressure."),
    
    Document(page_content="Sleeping 7–9 hours every night is vital for body recovery and overall health. "
                          "I love pizza, but eating it late at night can affect sleep cycles. "
                          "Good sleep helps improve memory and immune system strength."),
    
    # Irrelevant document
    Document(page_content="The Eiffel Tower is in Paris, and it attracts millions of tourists every year. "
                          "Photography enthusiasts often visit early in the morning for better lighting.")
]

# Create embeddings & FAISS vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

# Simple retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Compression retriever
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
)

# Query
query = "What are some ways to stay healthy?"
results = compression_retriever.get_relevant_documents(query)

print("\n--- Contextual Compression Results ---")
for r in results:
    print("-", r.page_content)
