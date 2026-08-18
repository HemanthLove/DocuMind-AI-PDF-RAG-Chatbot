# DocuMind AI – PDF RAG Chatbot

An AI-powered PDF chatbot that allows users to upload a PDF document, create a vector database from its content, and ask questions based on the uploaded document.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the document and generate context-aware answers.

---

## Features

- Upload PDF documents directly from the user interface
- Create a vector database from the uploaded PDF
- Split PDF content into smaller chunks
- Generate embeddings using a local Hugging Face embedding model
- Store embeddings in ChromaDB
- Retrieve relevant document chunks using MMR search
- Generate context-aware answers using Mistral AI
- Temporary chat history during the current session
- Simple step-by-step workflow
- Interactive Streamlit user interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face
- Sentence Transformers
- Mistral AI
- PyPDF
- python-dotenv

---

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

### 1. Upload PDF

The user uploads a PDF file through the Streamlit interface.

### 2. Load the PDF

The PDF content is loaded and processed using LangChain's `PyPDFLoader`.

### 3. Split the Document

The document is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

    chunk_size = 2000
    chunk_overlap = 200

### 4. Create Embeddings

Each text chunk is converted into a numerical vector using the following Hugging Face embedding model:

    sentence-transformers/all-MiniLM-L6-v2

The embedding model runs locally after being downloaded.

### 5. Store Embeddings in ChromaDB

The generated embeddings are stored in a ChromaDB vector database.

The vector database allows the application to search for the most relevant parts of the document when a user asks a question.

### 6. Retrieve Relevant Context

When the user enters a question, the application searches the vector database using **Maximum Marginal Relevance (MMR)**.

Current retrieval configuration:

    search_type = "mmr"
    k = 4
    fetch_k = 10
    lambda_mult = 0.5

### 7. Generate the Answer

The relevant document chunks, chat history, and user's question are sent to Mistral AI.

The model is instructed to answer using only the retrieved document context.

If the answer is not available in the document, the chatbot responds:

    I couldn't find the answer in the document.

---

## Application Flow

    Upload PDF
        ↓
    Load PDF Content
        ↓
    Split into Text Chunks
        ↓
    Generate Hugging Face Embeddings
        ↓
    Store Embeddings in ChromaDB
        ↓
    User Asks a Question
        ↓
    Retrieve Relevant Chunks using MMR
        ↓
    Combine Context + Chat History + Question
        ↓
    Generate Answer using Mistral AI

---

## Project Structure

    DocuMind-AI-PDF-RAG-Chatbot/
    │
    ├── app.py
    ├── create_database.py
    ├── main.py
    ├── requirements.txt
    ├── .gitignore
    │
    ├── document loaders/
    │
    └── Vector store/
        └── DB.py

The `chroma_db` folder is generated locally when a PDF is processed and is excluded from GitHub.

---

## Installation

### 1. Clone the Repository

    git clone https://github.com/HemanthLove/DocuMind-AI-PDF-RAG-Chatbot.git

### 2. Move into the Project Folder

    cd DocuMind-AI-PDF-RAG-Chatbot

### 3. Create a Virtual Environment

    python -m venv .venv

### 4. Activate the Virtual Environment

For Windows PowerShell:

    .venv\Scripts\Activate.ps1

For Windows Command Prompt:

    .venv\Scripts\activate

### 5. Install Dependencies

    python -m pip install -r requirements.txt

---

## Environment Variables

Create a `.env` file in the project folder.

Add your Mistral API key:

    MISTRAL_API_KEY=your_mistral_api_key_here

Make sure your `.env` file is included in `.gitignore`.

Example:

    .env
    .venv/
    chroma_db/
    __pycache__/

Never upload your API keys to GitHub.

---

## Run the Application

Run the Streamlit application using:

    streamlit run app.py

If `streamlit` is not recognized, use:

    python -m streamlit run app.py

The application will then open in your browser.

---

## How to Use

### Step 1: Upload a PDF

Open the application and select a PDF file from your computer.

### Step 2: Create the Vector Database

Click the button to create the vector database.

The application will:

- Read the PDF
- Split the document into chunks
- Generate embeddings
- Store the embeddings in ChromaDB

### Step 3: Ask Questions

Once the vector database has been created, enter a question related to the uploaded PDF.

For example:

    What is a Deep learning?

or:

    Explain Activation Function.

The application retrieves the most relevant information from the document and generates a context-aware answer.

---

## Chat Memory

The application maintains temporary chat history during the current session.

This allows the chatbot to remember previous messages and use them as conversational context.

The chat history is not permanently stored and is cleared when the application session ends or restarts.

---

## Important Notes

- A Mistral API key is required for generating AI responses.
- The Hugging Face embedding model runs locally.
- The embedding model may take some time to download during its first use.
- The ChromaDB vector database is created locally after processing a PDF.
- The vector database is not uploaded to GitHub.
- The quality of answers depends on the information available in the uploaded PDF.
- API keys should always be stored in the `.env` file.

---

## Future Improvements

- Support for multiple PDF files
- Source citations with page numbers
- Persistent chat history
- Conversation export
- Multiple embedding model options
- Support for different LLM providers
- Document management
- Improved chat interface
- Cloud deployment
- Support for additional document formats

---

## Author

**Hemanth Love**

GitHub: https://github.com/HemanthLove

LinkedIn: https://www.linkedin.com/in/hemanth-love/

---

## License

This project is created for learning and portfolio purposes.

Feel free to explore, modify, and improve the project.
