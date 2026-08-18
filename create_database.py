# load pdf
# split into chunks 
# create the embeddings
# store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)
print("Total chunks:", len(chunks))

# chunks = chunks[:50]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = 'chroma_db'
)
print("Database created successfully!")
print("Stored chunks:", len(chunks))