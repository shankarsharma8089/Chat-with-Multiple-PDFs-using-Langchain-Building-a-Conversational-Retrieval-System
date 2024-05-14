import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import apikey

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = apikey.APIKEY

source_folder = ""

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# Create Streamlit UI
st.title("Doc ConversationBOT")

# Initialize chat_history as empty list
st.session_state.setdefault('chat_history', [])

query = st.text_input("Enter your query:")

# Initialize loaders
loaders = []

# Load PDF files
for filename in os.listdir(source_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(source_folder, filename)
        try:
            loader = PyPDFLoader(file_path)
            loaders.append(loader)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# Create or update index
if loaders:
    index_creator = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}) if PERSIST else VectorstoreIndexCreator()
    index = index_creator.from_loaders(loaders)

    # Placeholder for accessing vectors
    # Replace this with the correct method or attribute to access vectors
    vectors = None

    # Displaying embeddings
    if st.checkbox("View Embeddings"):
        st.write("Embeddings:", vectors)

    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    # Perform query and display result
    if query:
        result = chain({"question": query, "chat_history": st.session_state.chat_history})
        st.write("Answer:", result['answer'])
        st.session_state.chat_history.append((query, result['answer']))
else:
    st.write("No PDF files found in the source folder.")

# Button to show/hide conversation history
if st.sidebar.button("Show Conversation History"):
    st.sidebar.subheader("Conversation History")
    for i, (query, answer) in enumerate(st.session_state.chat_history):
        st.sidebar.text(f"{i+1}. User: {query}")
        st.sidebar.text(f"   Bot: {answer}")

# Button to reset conversation history
if st.sidebar.button("Reset Conversation"):
    st.session_state.chat_history.clear()  # Clear the conversation history
