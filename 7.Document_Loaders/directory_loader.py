from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os

books_dir = os.path.join(os.path.dirname(__file__), 'books')
loader = DirectoryLoader(
    path = books_dir,
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)
docs = loader.lazy_load()

docs = list(docs)
# docs = loader.load()
# print(len(docs))
for documents in docs:
    print(docs[2].page_content)