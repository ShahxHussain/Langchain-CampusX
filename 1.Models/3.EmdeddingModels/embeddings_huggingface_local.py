from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = "E:/huggingface_cache"


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",

)

# text = "Pakistan is a country in Asia."
# vector = embeddings.embed_query(text)
# print(str(vector))


documents = [
    "Pakistan is in asia",
    "Islamabad is the capital of Pakistan",
    "Pakistan is a democratic country",
    "Pakistan is a nuclear power",
    "Paris is the capital of France",
    "Delhi is the capital of india"
]

embeddings = embedding.embed_documents(documents)
print(embeddings)