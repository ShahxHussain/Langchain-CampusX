from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)
model2 = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)

#1st prompt: Detailed Report
template1 = PromptTemplate(
    template =  "Write a short and simple notes on the following text \n {text}",
    input_variables= ['text']
)
#2nd prompt: Summary of result 1
template2 = PromptTemplate(
    template= "Provide a 5 short answer question on the following \n  {text}",
    input_variables= ['text']
)

template3 = PromptTemplate(
    template= "Merge the provided notes and quiz in a single document \n NOTES: {notes}\n QUIZ: {quiz}",
    input_variables= ['notes', 'quiz']
)

text = """
An AI engineer is a professional who designs, builds, and maintains AI systems and applications. 
They leverage machine learning, deep learning, and natural language processing to create intelligent solutions. 
Unlike researchers who focus on theoretical advancements, 
AI engineers focus on practical applications and deployment of AI technologies.
"""

parser = StrOutputParser()

Parallel_chain = RunnableParallel({
    'notes': template1 | model1 | parser,
    'quiz': template2 | model2 | parser
})

merge_chain = template3 | model1 | parser
chain = Parallel_chain | merge_chain
result = chain.invoke({'text': text})
print(result)

# Visualize Chain
# chain.get_graph().print_ascii()