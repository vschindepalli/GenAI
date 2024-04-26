import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import io
from openai import OpenAI
import anthropic
import faiss
from PIL import Image

st.set_page_config(page_title="LLM Dashboard with Dynamic RAG Feature")

# Sidebar
st.sidebar.title("Interaction Options")
interaction_option = st.sidebar.radio("Choose Interaction Mode", ("Interact with Uploaded Document", "Direct Interaction with Model"))

# PDF upload section (only visible if the user chooses to interact with the uploaded document)
uploaded_pdf = None
if interaction_option == "Interact with Uploaded Document":
    st.sidebar.header("Upload PDF Document")
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Model selection
st.sidebar.title("Model Selection")
option = st.sidebar.selectbox('Choose Your Model', ('OpenAI GPT-4', 'Claude 3 Opus'))

# Main content
st.title("LLM Dashboard with Dynamic RAG Feature")

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
    global faiss_index, tfidf_vectorizer
    vector = text_to_vector(text, fit_new_data=create_new_index)
    if create_new_index:
        # Get the dimensionality of the vector
        dimension = len(tfidf_vectorizer.vocabulary_)
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



# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

# Chat interface for direct interaction with the model
st.subheader("Chat Interface")

# Chat interface for direct interaction with the model
if interaction_option == "Direct Interaction with Model":
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    # Display previous chat messages
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    # Input bar at the bottom
    if prompt := st.chat_input():
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if option == "OpenAI GPT-4":
            try:
                # Query using OpenAI's GPT-4
                client = OpenAI(api_key="your_api_key")  # Replace with your OpenAI API key
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        elif option == "Claude 3 Opus":
            try:
                # Query using Claude 3 Opus
                claude_api_key = "your_api_key"  # Replace with your Claude API key
                claude_model = "claude-3-opus-20240229"
                claude_client = anthropic.Anthropic(api_key=claude_api_key)
                claude_response = claude_client.messages.create(
                    model=claude_model,
                    max_tokens=1000,
                    temperature=temperature,
                    system=prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                raw = claude_response.content
                response = raw[0].text
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"An error occurred with Claude: {e}")


# RAG capabilities
if interaction_option == "Interact with Uploaded Document" and uploaded_pdf is not None:
    # Process the uploaded file
    if uploaded_pdf.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_pdf)
    else:  # Assuming text file
        text = uploaded_pdf.getvalue().decode("utf-8")

    # Fit the vectorizer with the uploaded document text and create the FAISS index
    add_document_to_faiss(text, create_new_index=True)

    # Store the uploaded document text in session state
    st.session_state['uploaded_text'] = text
    st.write("Document uploaded and processed. You can now query about this document.")

# Query about the uploaded document
if interaction_option == "Interact with Uploaded Document" and 'uploaded_text' in st.session_state:
    user_query = st.text_input("Enter your query about the uploaded document:")
    if st.button("Retrieve Information"):
        retrieved_doc_index = retrieve_and_generate(user_query)
        document_sections = st.session_state['uploaded_text'].split('\n\n')  # Example split by double newlines
        response_text = document_sections[retrieved_doc_index] if retrieved_doc_index < len(document_sections) else "No relevant section found."

        if option == "OpenAI GPT-4":
            try:
                client = OpenAI(api_key = "your_api_key")
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": response_text},
                        {"role": "user", "content": user_query}
                    ]
                )
                st.write(completion.choices[0].message.content)  # Display the response from OpenAI
            except Exception as e:
                st.error(f"An error occurred with OpenAI: {e}")
        elif option == "Claude 3 Opus":
            try:
                claude_api_key = "your_api_key"  # Replace with your Claude API key
                claude_client = anthropic.Anthropic(api_key=claude_api_key)
                claude_model = "claude-3-opus-20240229"
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
