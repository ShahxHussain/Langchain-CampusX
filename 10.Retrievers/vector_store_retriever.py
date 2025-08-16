from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(page_content="LangChain is a framework for building applications powered by large language models (LLMs).")
doc2 = Document(page_content="FAISS is a library for fast similarity search and clustering of dense vectors developed by Meta AI.")
doc3 = Document(page_content="Generative AI (GenAI) creates new content such as text, images, audio, or code using machine learning models.")
doc4 = Document(page_content="LLMs (Large Language Models) are trained on massive datasets to understand and generate human-like text.")
doc5 = Document(page_content="LangChain provides modular tools like chains, agents, memory, and retrievers for AI workflows.")
doc6 = Document(page_content="FAISS supports indexing methods like flat indexes, inverted files, and HNSW for scalable vector search.")
doc7 = Document(page_content="GenAI powers chatbots, text summarizers, code generators, and content creation tools.")
doc8 = Document(page_content="LLMs can be fine-tuned with methods like LoRA, PEFT, or instruction-tuning for domain-specific tasks.")
doc9 = Document(page_content="LangChain integrates with vector databases like Pinecone, Weaviate, FAISS, and Milvus for RAG.")
doc10 = Document(page_content="Combining LangChain, FAISS, and LLMs enables efficient retrieval and contextual AI applications.")

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = FAISS.from_documents(
    docs,
    embeddings_model
)

# Enable MMR as retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

query = "What is LLM?"
result = retriever.invoke(query)


for doc in result:
    print(doc.page_content)