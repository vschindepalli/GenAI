import streamlit as st
import pdfplumber
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Title of the app
st.title("Mistral RAG")

# File uploader
with st.sidebar.header("1. Upload your file"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["txt", "pdf", "py"])

# Text area for user input
user_input = st.text_area("Enter your question")

# Button to trigger the model
button_clicked = st.button("Get Answer")

# Load data
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            docs = '\n'.join(pages)
    else:
        loader = TextLoader(uploaded_file)
        docs = loader.load()
    
    # Create a Document object from the text
    doc = Document(page_content=docs)
    
    # Split text into chunks 
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents([doc])
    
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="your_api_key")
    
    # Create the vector store 
    vector = FAISS.from_documents(documents, embeddings)
    
    # Define a retriever interface
    retriever = vector.as_retriever()
    
    # Define LLM
    model = ChatMistralAI(model = "open-mixtral-8x22b",mistral_api_key="your_api_key")
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    
    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get the answer
    if button_clicked:
        response = retrieval_chain.invoke({"input": user_input})
        st.write(response["answer"])