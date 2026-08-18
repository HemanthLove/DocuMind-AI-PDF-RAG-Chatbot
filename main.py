from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
chat_history = []

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory = "chroma_db",
    embedding_function = embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs ={
        "k":4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatMistralAI(model ="mistral-small-2603")

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [ 
        (
            "system",
            """You are a helpful AI assistant.
            Use only the provided context to answer the question.
            If the answer is not present in the context, say "I couldn't find the answer in the document."
            """
        ),
        (
            'human', """
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

print("RAG system created")
print("press 0 to exit")
while True:
    query = input("You: ")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    history_text = "\n".join(
        [
            f"User: {item['user']}\nAI: {item['ai']}"
            for item in chat_history
        ]
    )

    final_prompt = prompt.invoke({
        "chat_history": history_text,
        "context": context,
        "question": query
    })
                                 
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")

    chat_history.append({
        "user": query,
        "ai": response.content
    })