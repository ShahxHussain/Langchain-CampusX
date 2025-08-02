from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.2-3B-Instruct",
    task = "text-generation",

)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is the founder of Pakistan?")
print(result.content)