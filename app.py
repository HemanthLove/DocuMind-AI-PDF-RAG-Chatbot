import streamlit as st
import tempfile
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()


# Streamlit page settings
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# -------------------------------
# SESSION STATE
# -------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "database_created" not in st.session_state:
    st.session_state.database_created = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# TITLE
# -------------------------------

st.title("📚 PDF RAG Chatbot")

st.write("Upload a PDF → Create Vector Database → Ask Questions")


# -------------------------------
# STEP 1: UPLOAD PDF
# -------------------------------

st.header("Step 1: Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)


# -------------------------------
# STEP 2: CREATE DATABASE
# -------------------------------

st.header("Step 2: Create Vector Database")


if uploaded_file is not None:

    if st.button("Create Vector Database"):

        with st.spinner("Creating vector database..."):

            # Reset old database
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")


            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name


            # Load PDF
            loader = PyPDFLoader(temp_path)

            documents = loader.load()


            # Split PDF into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)


            # Create embedding model
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )


            # Create Chroma vector database
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )


            # Store vector database in session
            st.session_state.vectorstore = vectorstore

            st.session_state.database_created = True

            st.session_state.messages = []


            # Delete temporary PDF
            os.unlink(temp_path)


        st.success("Vector Database Created Successfully!")

        st.write(f"📄 PDF: {uploaded_file.name}")

        st.write(f"🧩 Total Chunks: {len(chunks)}")

        st.info("You can now move to Step 3 and ask questions.")


else:

    st.warning("Please upload a PDF first.")


# -------------------------------
# STEP 3: ASK QUESTIONS
# -------------------------------

st.header("Step 3: Ask Questions")


if not st.session_state.database_created:

    st.info("Create the vector database first to start chatting.")


else:

    # Show chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


    query = st.chat_input(
        "Ask a question about your PDF..."
    )


    if query:

        # Show user message
        with st.chat_message("user"):

            st.write(query)


        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })


        with st.chat_message("assistant"):

            with st.spinner("Searching the document..."):

                # Create retriever
                retriever = st.session_state.vectorstore.as_retriever(

                    search_type="mmr",

                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }

                )


                # Retrieve relevant documents
                docs = retriever.invoke(query)


                # Combine retrieved chunks
                context = "\n\n".join(

                    [doc.page_content for doc in docs]

                )


                # Create chat history
                history_text = ""


                for message in st.session_state.messages[:-1]:

                    if message["role"] == "user":

                        history_text += f"User: {message['content']}\n"

                    else:

                        history_text += f"AI: {message['content']}\n"


                # Prompt
                prompt = ChatPromptTemplate.from_messages(

                    [

                        (
                            "system",

                            """
                            You are a helpful AI assistant.

                            Answer the question using only the provided
                            document context.

                            Use the chat history only to understand
                            follow-up questions.

                            If the answer is not present in the document,
                            say:

                            "I couldn't find the answer in the document."
                            """

                        ),

                        (

                            "human",

                            """
                            Chat History:
                            {chat_history}

                            Context:
                            {context}

                            Question:
                            {question}
                            """

                        )

                    ]

                )


                # Load Mistral LLM
                llm = ChatMistralAI(

                    model="mistral-small-2603"

                )


                # Create final prompt
                final_prompt = prompt.invoke({

                    "chat_history": history_text,

                    "context": context,

                    "question": query

                })


                # Get response
                response = llm.invoke(final_prompt)


                answer = response.content


                # Display answer
                st.write(answer)


        # Save AI response to chat history
        st.session_state.messages.append({

            "role": "assistant",

            "content": answer

        })