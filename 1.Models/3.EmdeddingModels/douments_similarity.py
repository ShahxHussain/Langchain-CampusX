from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Pakistan is in asia",
    "Islamabad is the capital of Pakistan",
    "Pakistan is a democratic country",
    "Pakistan is a nuclear power",
    "Paris is the capital of France",
    "Delhi is the capital of india"
]

query = "What is the capital of pakistan?"

doc_embeddings = embedding_model.embed_documents(documents)
query_embedding = embedding_model.embed_query(query)

# print(doc_embeddings)
# print(query_embedding)

# print(cosine_similarity([query_embedding], doc_embeddings)) # this will print the similarity between the query and the documents

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# this will print the index and the similarity score
# for i, score in enumerate(scores):
#     print(f"Document {i}: {score}")

index , score = (sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]) # this will print the index and the similarity score in descending order

print(query)
print(documents[index])
print("Similarity Score is: ", score)






