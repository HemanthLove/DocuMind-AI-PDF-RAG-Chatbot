from langchain_chroma import Chroma

from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Aritificial Intelligence. ", metadata={"source":"AI Book"}),
    Document(page_content="Pandas is used for data analytics in python. ", metadata={"source":"Data science book"}),
    Document(page_content="Neural networks are used in deep learning. ", metadata={"source":"DLbook"}),
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding = embedding_model,
    persist_directory= "chroma-db"
)

result = vectorstore.similarity_search("What is used for data analysis?", k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")
for d in docs:
    print(d.page_content)
    print(d.metadata)