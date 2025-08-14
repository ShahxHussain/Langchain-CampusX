from langchain.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer

# Function to split text semantically using LangChain

def split_text_semantically(text):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_splitter = SemanticChunker(embedding_model)
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    sample_text = """
    Dogs are amazing animals.
    Cats are often considered the easiest pets to care for.
    Advances in artificial intelligence have led to sophisticated robots.
    Space exploration is increasingly reliant on AI technologies.
    """
    result = split_text_semantically(sample_text)
    for i, chunk in enumerate(result):
        print(f"Chunk {i+1}: {chunk}\n")