from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template = """ 
        Paper Name: {paper_input}
        Explanation Style: {style_input}
        Explanation Length: {length_input}
        Act as a paper reviewer and explain the paper in a way Best way to understand the paper.
        Analogies: use Ralatable Analogies to simplify the complex ideas
        NOTE: If certain informtation is not available in the paper, don't use your own knowledge respond with "I don't know".
"""
,

input_variables = ["paper_input", "style_input", "length_input"]

)


template.save("template.json")