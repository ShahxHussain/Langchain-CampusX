from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

st.header ("Research Tool UI + Prompt Practice")

paper_input = st.selectbox("Select the paper Name", ["Attention is all you need",
                                                      "BERT: Pre Training of Deep Bidirectional Transformers for Language Understanding",
                                                      "GPT-3: Langauage models are few shot leaners"])


style_input = st.selectbox("Select Explanation style", ["Beginner Friendly",
                                                        "Technical", 
                                                        "Code Oriented", 
                                                        "Mathematical"])       

length_input = st.selectbox("Select Explanation length", ["1- 3 Paragraph", 
                                                          "4- 5 Paragraph", 
                                                          "Detail Explanation"])

'''
Methond 1
'''
# template = PromptTemplate(
#     template = """ 
#         Paper Name: {paper_input}
#         Explanation Style: {style_input}
#         Explanation Length: {length_input}
#         Act as a paper reviewer and explain the paper in a way Best way to understand the paper.
#         Analogies: use Ralatable Analogies to simplify the complex ideas
#         NOTE: If certain informtation is not available in the paper, don't use your own knowledge respond with "I don't know".
# """
# ,

# input_variables = ["paper_input", "style_input", "length_input"]

# )

'''
Methond 2
'''
template = load_prompt("template.json")

user_input = st.text_input("Enter your age")

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button("Summarzie"):
    result = model.invoke(prompt)
    st.write(result.content) # for UI output
    print(result.content) # for terminal output



# Prompt in chain

# if st.button("Summarzie"):
#     chain = model | template
#     result = chain.invoke({
#         'paper_input': paper_input,
#         'style_input': style_input,
#         'length_input': length_input
#     })

#     st.write(result.content) # for UI output
#     print(result.content) # for terminal output






