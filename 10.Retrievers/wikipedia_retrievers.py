from langchain_community.retrievers import WikipediaRetriever

#intialize the retriever
retriever = WikipediaRetriever(top_k_results= 2, lang="en")

query =  "Who is Muhammad Ali Jinnah?"

#get Relevant info
docs = retriever.invoke(query)

print(docs)


for i, document in enumerate(docs):
    print("Result", i+1)
    print("Content: \n", document.page_content)