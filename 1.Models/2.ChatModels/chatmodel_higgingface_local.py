from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

os.environ['HF_HOME'] = "E:/huggingface_cache"

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs = dict(
        max_tokens = 100,
        temperature = 0.5,
    )
)
model = ChatHuggingFace(llm = llm)

result = model.invoke("Who is the founder of Pakistan?")
print(result.content)