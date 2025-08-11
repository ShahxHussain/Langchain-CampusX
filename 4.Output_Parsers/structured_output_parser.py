from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)
schema = [
    ResponseSchema(name = 'area', description = 'Area of a country'),
    ResponseSchema(name = 'major_religion', description = 'Major religion of a country'),
    ResponseSchema(name = 'capital', description = 'Capital of a country'),
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template= "Give me the the area, major_religion, and capital of the {country}\n {format_instructions}",
    input_variables=['country'],
    partial_variables= {'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'country': 'Pakistan'})
print(result)
