import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

st.markdown("""
<style>
    /* Global styles */
    body {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main container */
    .main {
        background-color: #121212; /* Darker background for a modern look */
        color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
    }

    /* Text input styling */
    .stTextInput textarea {
        color: #ffffff !important;
        background-color: #2d2d2d !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .stTextInput textarea:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 8px rgba(0, 255, 136, 0.4) !important;
    }

    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #00ff88 !important;
    }

    .stSelectbox svg {
        fill: white !important;
        transition: transform 0.3s ease;
    }

    .stSelectbox:hover svg {
        transform: rotate(180deg);
    }

    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }

    /* Dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
        padding: 10px;
        border-radius: 8px;
        margin: 4px 0;
        transition: background-color 0.3s ease;
    }

    div[role="listbox"] div:hover {
        background-color: #00ff88 !important;
        color: #121212 !important;
    }

    /* Button styling */
    .stButton button {
        background-color: #00ff88 !important;
        color: #121212 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }

    .stButton button:hover {
        background-color: #00cc66 !important;
        transform: scale(1.05);
    }

    /* Add a glowing effect to interactive elements */
    .stTextInput textarea, .stSelectbox div[data-baseweb="select"], .stButton button {
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }

    ::-webkit-scrollbar-thumb {
        background: #00ff88;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #00cc66;
    }
</style>
""", unsafe_allow_html=True)
st.title("Coding Buddy")
st.caption("Your AI programming and debugging buddy!")

# Sidebar configuration
with st.sidebar:
    # Header with a cool icon and gradient text
    st.markdown("""
    <h2 style="color: #00ff88; background: linear-gradient(90deg, #00ff88, #00cc66); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline;">
        ⚙️ Configuration
    </h2>
    """, unsafe_allow_html=True)

    # Model selection dropdown with a sleek design
    selected_model = st.selectbox(
        "**Choose Model**",
        ["deepseek-r1:1.5b", "deepseek-r1:8b"],
        index=0,
        help="Select the model you want to use for your tasks."
    )

    # Divider with a glowing effect
    st.markdown("""
    <style>
        .stDivider > div {
            border-top: 2px solid #00ff88;
            box-shadow: 0 0 8px rgba(0, 255, 136, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)
    st.divider()

    # Model Capabilities section with icons and better spacing
    st.markdown("""
    <h3 style="color: #00ff88; margin-bottom: 16px;">
        🚀 Model Capabilities
    </h3>
    <div style="margin-left: 16px;">
        <p style="margin: 8px 0; font-size: 14px;">🐍 <strong>Python Genius</strong> - Write and optimize Python code like a pro.</p>
        <p style="margin: 8px 0; font-size: 14px;">🐞 <strong>Debugging Assistant</strong> - Find and fix bugs effortlessly.</p>
        <p style="margin: 8px 0; font-size: 14px;">📝 <strong>Code Documentation</strong> - Generate clear and concise documentation.</p>
        <p style="margin: 8px 0; font-size: 14px;">💡 <strong>Solution Design</strong> - Architect robust and scalable solutions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Optional: Add a small footer or note
    st.markdown("""
    <div style="margin-top: 24px; font-size: 12px; color: #777;">
        Powered by DeepSeek 🤖
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/) | [DeepSeek](https://chat.deepseek.com/)")

# Main content area + chat engine

llm = ChatOllama(
    model=selected_model,
    base_url = "http://localhost:11434",
    temperature = 0.5,
)

# Define the system message
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful programming assistant. Provide concise and accurate answers to the user's coding questions. You are also knowledgable in debugging so help the user debug using good techniques."
)

#Session state management
if "generated" not in st.session_state:
    st.session_state["generated"] = [{"role": " ai assistant", "content": "🤖Hi! I'm your coding assistant powered by DeepSeek. Ask me anything related to programming and I'll do my best to assist you!"}]

#Displaying the convo
chat_container = st.container()
with chat_container:
    for message in st.session_state["generated"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_input = st.chat_input("Ask your questiom here!")

def get_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for text in st.session_state.generated:
        if text["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(text["content"]))
        elif text["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(text["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_input:
    # Store user query in the session state
    st.session_state.generated.append({"role": "user", "content": user_input})
    # Generate AI response
    with(st.spinner("Generating response...")):
        prompt_chain = build_prompt_chain()
        ai_response = get_ai_response(prompt_chain)
    # Store AI response in the session state
    st.session_state.generated.append({"role": "ai", "content": ai_response})
    # Clear the input box
    st.rerun()
