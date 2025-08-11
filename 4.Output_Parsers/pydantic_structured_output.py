from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

# Define Pydantic schema for country information
class CountryInfo(BaseModel):
    name: str = Field(description="Name of the country")
    capital: str = Field(description="Capital city of the country")
    population: int = Field(description="Population of the country in millions")
    area_km2: float = Field(description="Area of the country in square kilometers")
    major_religions: List[str] = Field(description="List of major religions practiced in the country")
    continent: Literal["Asia", "Europe", "Africa", "North America", "South America", "Australia", "Antarctica"] = Field(description="Continent where the country is located")
    gdp_per_capita: Optional[float] = Field(description="GDP per capita in USD", default=None)
    languages: List[str] = Field(description="Official and major languages spoken in the country")
    timezone: str = Field(description="Primary timezone of the country")
    currency: str = Field(description="Official currency of the country")

parser = PydanticOutputParser(pydantic_object=CountryInfo)

template = PromptTemplate(
    template= "Generate all the required info about {country} \n {format_instructions}",
    input_variables=['country'],
    partial_variables= {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'country': 'Pakistan'})
print(result)