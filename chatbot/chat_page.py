import streamlit as st
from chatbot import Chatbot
import os


# system= "You are a smart research assistant. It is important that you attempt to to utilize the given documents in the database to retrieve information first, via the retrieve_documents tool. If you are unable to find any relevant information, you can utilize the TavilySearch tool. You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that! """

system= """You are a helpful assistant with the main purpose of answering queries related to the documents you are given. You have access to a document database and web search, if the documents don't contain the information needed.

IMPORTANT: Always try to answer questions using the retrieve_documents tool FIRST, 
as it searches your internal knowledge base. Only use web search (tavily) if:
- The question requires current/real-time information
- The document database doesn't contain relevant information
- The user explicitly asks for web search

You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that! """


@st.cache_resource



def get_chatbot():
    return Chatbot(system=system)

chatbot = get_chatbot()

def get_filepaths(uploaded_files):
    filepaths = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with open(f"./temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        filepaths.append(f"./temp_{uploaded_file.name}")
 
    return filepaths

# Page configuration
st.set_page_config(page_title="Chatbot Interface", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"

# Custom CSS to style the chat container with a black border and other UI elements
st.markdown("""
<style>
    /* Style for the chat box container */
    .chat-box {
        padding: 20px;
        height: 500px;
        overflow-y: auto;
        background-color: #ffffff;
        margin-bottom: 20px;
    }
    
    /* Style for individual messages */
    .user-message {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
        display: inline-block;
        float: right;
        clear: both;
        max-width: 70%;
    }
    
    .bot-message {
        background-color: #e6e9ef;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
        display: inline-block;
        float: left;
        clear: both;
        max-width: 70%;
    }

    /* Clear float fix */
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    
    /* Hide default Streamlit main menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Sidebar for navigation and data management

with st.sidebar:
    st.markdown("---")
    st.subheader("Manage Data")


    
    # 1. Get the current state of the widget
    uploaded_files_list = st.file_uploader("", accept_multiple_files=True)
    
    # Initialize processed_files tracker if it doesn't exist
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # --- LOGIC A: Add New Files ---
    if uploaded_files_list:
        # Find files that are in the widget but NOT in our processed set
        new_files = [f for f in uploaded_files_list if f.name not in st.session_state.processed_files]

        if new_files:
            filepaths = get_filepaths(new_files)
            
            # Uncomment when chatbot is connected
            chatbot.load_embeddings(filepaths)
            for file_path in filepaths:
                os.remove(file_path)
            print("File deleted successfully")            
            for f in new_files:
                st.session_state.processed_files.add(f.name)
            
            st.success(f"Added {len(new_files)} new document(s)")


    # --- LOGIC B: Remove Deleted Files ---
    # Find files that are in our processed set but NO LONGER in the widget

    st.subheader("Database Contents")

    # Initialize session state with DB contents if not already done
    if "db_files" not in st.session_state:
        # This pulls the "truth" from Chroma when the app starts
        st.session_state.db_files = chatbot.get_existing_documents()

    # Display existing files
    if st.session_state.db_files:
        st.write(f"{len(st.session_state.db_files)} documents indexed")
        
        # Create a list of files to delete
        files_to_delete = []
        
        # Display each file with a checkbox or trash icon
        for file in list(st.session_state.db_files):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text(file)
            with col2:
                # Unique key is needed for every button
                if st.button("x", key=f"del_{file}", help=f"Delete {file}"):
                    files_to_delete.append(file)
        
        # Process Deletion
        if files_to_delete:
            for file in files_to_delete:
                success = chatbot.delete_document(file)
                if success:
                    st.session_state.db_files.remove(file)
                    st.toast(f"Deleted {file}", icon="🗑️")
                else:
                    st.error(f"Could not delete {file}")
            
            # Rerun to update the list visually
            st.rerun()
            
    else:
        st.info("Database is empty.")



# Main Content Area
st.title(st.session_state.current_page)

if st.session_state.current_page == "Chat":
    # 1. Display Chat History in a styled box
    chat_container = st.container()
    
    # We use a markdown div to create the 'black border box' visual
    # Inside, we render the messages
    message_html = '<div class="chat-box">'
    
    if not st.session_state.messages:
        message_html += '<p style="color: grey; text-align: center; margin-top: 200px;">Start a conversation...</p>'
    
    for role, text in st.session_state.messages:
        if role == "user":
            message_html += f'<div class="clearfix"><div class="user-message">👤 <b>You:</b><br>{text}</div></div>'
        else:
            message_html += f'<div class="clearfix"><div class="bot-message">🤖 <b>Bot:</b><br>{text}</div></div>'
            
    message_html += '</div>'
    
    with chat_container:
        st.markdown(message_html, unsafe_allow_html=True)

    # 2. Input Area
    # Using a form ensures 'Enter' key works to submit
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Message", 
                placeholder="Type something...", 
                label_visibility="collapsed",
                key="input_text"
            )
            
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)

        if submit_button and user_input:
            # Add user message
            st.session_state.messages.append(("user", user_input))
            
            
            # Simulate bot response
            # In a real app, you would call your chatbot API here

            bot_response = chatbot.query_chatbot(user_input)
            st.session_state.messages.append(("bot", bot_response))
            
            st.rerun()


