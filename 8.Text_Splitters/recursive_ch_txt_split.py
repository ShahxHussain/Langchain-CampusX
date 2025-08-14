from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text = """
class Calculator:
    # Constructor: initializes values when object is created
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2

    # Method: adds the two numbers and returns the result
    def add(self):
        return self.num1 + self.num2

# Create object of the class
calc = Calculator(10, 5)

# Call method and store result
result = calc.add()

# Print result
print("The sum is:", result)

"""

splitter = RecursiveCharacterTextSplitter(
    # language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(chunks[0])