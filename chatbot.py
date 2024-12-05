import os
import pandas as pd
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document


with open(r'C:\Users\dries.faems\Biografie\Open AI coding.txt', 'r', encoding = 'utf-8') as file:
    api_key = str(file.readline())[1:-1]

# Set the environment variable for the OpenAI API key
os.environ['OPENAI_API_KEY'] = api_key

# ------------------------------------------------------------
# Set your OpenAI API key if not already in your environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Directory where Chroma will persist its database
PERSIST_DIRECTORY = "chroma_db"

st.title("WHU Alumni Database Chatbot")
st.markdown("Query the alumni database using embeddings-based semantic search.")

# Initialize embeddings first
embeddings = OpenAIEmbeddings()

# Load from disk
vectorstore = Chroma(
    collection_name="whu_alumni",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY
)
st.write("Loaded existing vectorstore from disk.")

# Query input
query = st.text_input("Ask a question about the alumni database (e.g. 'Who worked at Procter & Gamble?')")

if query:
    # Perform similarity search
    results = vectorstore.similarity_search(query, k=3)
    st.write("### Results:")
    if results:
        for res in results:
            st.write(f"**Name:** {res.metadata['Name']}")
            st.write(f" - Jahrgang: {res.metadata['Jahrgang']}")
            st.write(f" - Company: {res.metadata['Company']}")
            st.write(f" - Position: {res.metadata['Position']}")
            st.write(f" - Stadt, Land: {res.metadata['Stadt']}, {res.metadata['Land']}")
            st.write("---")
    else:
        st.write("No results found.")
