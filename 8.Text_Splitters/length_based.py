from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separator = ''
) 
text = """
    O land of five rivers and countless dreams,
Where the Himalayas stand as ancient guardians,
And the Arabian Sea whispers tales to the shore,
You were born in the cry for freedom,
Forged in the ink of vision and the blood of sacrifice.
From Karachi’s bustling ports to Hunza’s silent valleys,
You wear every season like a crown,
Golden fields in Punjab sway like prayers in motion,
Desert winds of Thar carry songs older than time,
And snow-kissed peaks cradle the moon at night.
"""

result = splitter.split_text(text)
print(result)