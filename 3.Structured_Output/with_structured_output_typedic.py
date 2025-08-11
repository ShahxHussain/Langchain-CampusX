'''
Typed Dictionary is only for Representation purpose
Validation can't be added in.
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
'''
You can also add Literals and optionals
sentiment:  Annotated[Literal["P", "S", "N"], "Retrun sentiment of a review either positive, negative or neutral"]
name: Annotated[Optional[list[str]], "Write down the name of a reviever"]
'''

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)

# Schema
class Review(TypedDict):
    summary:  Annotated[str, "A brief summary of a review"]
    # sentiment:  Annotated[str, "Retrun sentiment of a review either positive, negative or neutral"]
    sentiment:  Annotated[Literal["P", "S", "N"], "Retrun sentiment of a review either positive, negative or neutral"]
    name: Annotated[Optional[str], "Write down the name of a reviever"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""COMSATS University Islamabad, Abbottabad Campus (CUI Abbottabad), 
                                 stands out as one of the most vibrant and forward-thinking higher education institutions in Pakistan.
                                  Nestled in the scenic hills of Abbottabad, the campus not only offers a peaceful and inspiring 
                                 learning environment but also upholds strong academic and research standards that rival national 
                                 and international institutions.
                                 Reviewed by Google
""")
print(result)
# print(['sentiment'])
# print(['summary'])
# print(['name'])





