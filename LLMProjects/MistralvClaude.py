import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import io
import faiss
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import anthropic
import json
import ast

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Initialize an empty FAISS index (will be created dynamically)
faiss_index = None

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to convert text to vector using TF-IDF
def text_to_vector(text, fit_new_data=False):
    global tfidf_vectorizer
    if fit_new_data:
        # Fit the vectorizer on the text and transform it into a vector
        vector = tfidf_vectorizer.fit_transform([text]).toarray()[0]
    else:
        # Transform the text into a vector using the already fitted vectorizer
        vector = tfidf_vectorizer.transform([text]).toarray()[0]
    return vector

# Function to add document to FAISS index
def add_document_to_faiss(text, create_new_index=False):
    global faiss_index
    vector = text_to_vector(text, fit_new_data=create_new_index)
    if create_new_index:
        # Create a new FAISS index based on the dimensionality of the first vector
        dimension = vector.shape[0]
        faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array([vector]))

# Function to perform RAG using FAISS
def retrieve_and_generate(query):
    global tfidf_vectorizer
    # Check if the vectorizer has been fitted
    if not hasattr(tfidf_vectorizer, 'vocabulary_'):
        raise ValueError("The TF-IDF vectorizer is not fitted. Please upload a document first.")
    query_vector = text_to_vector(query)
    _, indices = faiss_index.search(np.array([query_vector]), k=1)
    return indices[0][0]

# Set up the Mistral client with your API key
mistral_api_key = "your_api_key"
mistral_model = "mistral-large-latest"
mistral_client = MistralClient(api_key=mistral_api_key)

# Set up the Claude client with your API key
claude_api_key = "your_api_key"  # Assuming the API key is set in this environment variable
claude_model = "claude-3-opus-20240229"
claude_client = anthropic.Anthropic(api_key=claude_api_key)

# Streamlit UI
st.title("LLM Dashboard with Dynamic RAG Feature")

# User selects the model to interact with
model_choice = st.radio("Choose a model to interact with:", ("Mistral Large", "Claude 3 Opus"))

# User uploads a document
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
if uploaded_file is not None:
    # Process the uploaded file
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:  # Assuming text file
        text = uploaded_file.getvalue().decode("utf-8")
    
    # Fit the vectorizer with the uploaded document text and create the FAISS index
    add_document_to_faiss(text, create_new_index=True)
    
    # Store the uploaded document text in session state
    st.session_state['uploaded_text'] = text
    st.write("Document uploaded and processed. You can now query about this document.")

# Query about the uploaded document
if 'uploaded_text' in st.session_state:
    user_query = st.text_input("Enter your query about the uploaded document:")
    if st.button("Retrieve Information"):
        retrieved_doc_index = retrieve_and_generate(user_query)
        document_sections = st.session_state['uploaded_text'].split('\n\n')  # Example split by double newlines
        response_text = document_sections[retrieved_doc_index] if retrieved_doc_index < len(document_sections) else "No relevant section found."
        
        if model_choice == "Mistral Large":
            try:
                chat_response = mistral_client.chat(
                    model=mistral_model,
                    messages=[
                        ChatMessage(role="system", content=response_text),
                        ChatMessage(role="user", content=user_query)
                    ]
                )
                mistral_response = chat_response.choices[0].message.content
                st.write(mistral_response)  # Display the response from Mistral
            except Exception as e:
                st.error(f"An error occurred with Mistral: {e}")
        else:
            try:
                
                claude_response = claude_client.messages.create(
                    model=claude_model,
                    max_tokens=1000,
                    temperature=0.0,
                    system=response_text,
                    messages=[
                        {"role": "user", "content": user_query}
                    ]
                )
                raw = claude_response.content
                response = raw[0].text
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred with Claude: {e}")

# Chat with the model
user_message = st.text_input("Enter your message to chat with the model:")
if st.button("Send Message"):
    if model_choice == "Mistral Large":
        # Send the user's message to Mistral and display the response
        chat_response = mistral_client.chat(
            model=mistral_model,
            messages=[ChatMessage(role="user", content=user_message)]
        )
        mistral_response = chat_response.choices[0].message.content
        st.write(mistral_response)  # Display the response from Mistral
    

    else:
        # Send the user's message to Claude and display the response
        claude_response = claude_client.messages.create(
            model=claude_model,
            max_tokens=1000,
            temperature=0.0,
            system="Respond as a normal human being.",
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
       
        raw = claude_response.content
        response = raw[0].text
        st.write(response)# Display the response from Claude