from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Mixed documents: Health + Irrelevant
doc1 = Document(page_content="Regular exercise improves cardiovascular health and strengthens muscles.")
doc2 = Document(page_content="A balanced diet with proteins, carbs, and healthy fats supports overall body health.")
doc3 = Document(page_content="Drinking enough water daily is essential for hydration and energy levels.")
doc4 = Document(page_content="Getting 7-9 hours of sleep helps in muscle recovery and mental well-being.")
doc5 = Document(page_content="Too much sugar intake increases the risk of obesity and diabetes.")
doc6 = Document(page_content="Fruits and vegetables provide essential vitamins, minerals, and antioxidants.")
doc7 = Document(page_content="Stress management techniques like meditation improve mental and physical health.")
doc8 = Document(page_content="Python is a popular programming language for data science and machine learning.")
doc9 = Document(page_content="The Eiffel Tower is one of the most famous landmarks in Paris.")
doc10 = Document(page_content="Mars is called the Red Planet because of its iron oxide-rich soil.")

# Irrelevant docs


docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]

# Embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Vector store
vector_store = FAISS.from_documents(
    docs,
    embeddings_model
)

# LLM for query expansion
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# MultiQueryRetriever with k=3
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)

# Simple retriever with k=3
simple_retriever = vector_store.as_retriever(search_type = "similarity" , search_kwargs={"k": 2})

# Query example
query = "How can I stay healthy?"


print("\n--- MultiQueryRetriever Results ---")
multi_results = multi_retriever.invoke(query)
for doc in multi_results:
    print("-", doc.page_content)

print("\n--- Simple Retriever Results ---")
simple_results = simple_retriever.invoke(query)
for doc in simple_results:
    print("-", doc.page_content)
