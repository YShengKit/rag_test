__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from pydantic import BaseModel, Field
from typing import Literal
from chromadb import PersistentClient
from langchain_community.document_loaders import PyPDFLoader

# Define the state of the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# User inputs for API keys
st.sidebar.header("API Keys")
google_api_key = st.sidebar.text_input("Google AI API Key", type="password")

# Set environment variables
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# User inputs for vector store selection
st.sidebar.header("Vector Store")
vector_store_option = st.sidebar.selectbox("Choose Vector Store", ["Chroma"])

# Initialize vector store client
if vector_store_option == "Chroma":
    client = PersistentClient(path="./chroma_langchain_db")
    collections = [col for col in client.list_collections()]
    st.sidebar.write(f"Current selected vector store: {vector_store_option}")  # Debug output

# User selects or creates a collection
st.sidebar.subheader("Collection")
collection_names = collections + ["Create New Collection"] if collections else ["Create New Collection"]
collection_option = st.sidebar.selectbox("Select Collection", collection_names)

if collection_option == "Create New Collection":
    new_collection_name = st.sidebar.text_input("New Collection Name")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")
    if st.sidebar.button("Create Collection") and new_collection_name and uploaded_file:
        
        st.sidebar.success(f"Collection '{new_collection_name}' created successfully!")
        collections.append(new_collection_name)
        collection_option = new_collection_name

        # Load and process the uploaded PDF
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

        all_splits = text_splitter.split_documents(docs)
        
        vector_store = Chroma(
            collection_name= new_collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )

        vector_store.add_documents(documents=all_splits)
        st.sidebar.write(f"Using collection: {collection_option}")  # Debug output
        st.sidebar.success(f"PDF processed into {len(all_splits)} document chunks!")
else:
    vector_store = Chroma(client=client, collection_name=collection_option, embedding_function=embeddings)
    st.sidebar.write(f"Using collection: {collection_option}")  # Debug output


# Define the nodes (application steps)
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search_with_score(state["question"],k=5)
    # Properly extract page_content and score
    context_with_scores = [
        {"text": doc.page_content, "score": score} for doc, score in retrieved_docs
    ]
    return {"context": context_with_scores}

def generate(state: State):
    docs_content = "\n\n".join(doc["text"] for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Define the control flow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize prompt
prompt = hub.pull("rlm/rag-prompt")

# User input for question
# question = st.text_input("Ask a question:")

# Process the question and display the answer
# if st.button("Submit"):
#     if not google_api_key:
#         st.error("Please enter your Google AI API Key.")
#     else:
#         result = graph.invoke({"question": question})
#         st.write(f"Context: {result['context']}")
#         st.write(f"Answer: {result['answer']}")


# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
st.title("Chat with Document RAG")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for question
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if not google_api_key:
        st.error("Please enter your Google AI API Key.")
    else:
        result = graph.invoke({"question": user_input})

        # Format context results with scores
        context_display = "\n\n".join(f"Scores: {ctx['score']:.4f}\n\n {ctx['text']}" for ctx in result["context"])

        with st.chat_message("assistant"):
            response = result["answer"]
            st.markdown(response)
            with st.expander("Retrieved Context (with Scores)"):
                st.markdown(context_display)

        st.session_state.messages.append({"role": "assistant", "content": response})
