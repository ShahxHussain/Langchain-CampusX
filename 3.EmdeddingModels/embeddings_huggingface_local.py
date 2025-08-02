from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = "E:/huggingface_cache"


embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",

)

text = "Pakistan is a country in Asia."

vector = embeddings.embed_query(text)
print(str(vector))