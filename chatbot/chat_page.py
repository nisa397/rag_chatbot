import streamlit as st
from chatbot import Chatbot
import os

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import logging
from rich.logging import RichHandler


system= """You are a helpful assistant with the main purpose of answering queries related to the documents you are given. You have access to a document database and web search to use if the documents don't contain the information needed.

IMPORTANT: Always try to answer questions using the retrieve_documents tool FIRST, 
as it searches your internal knowledge base. Only use web search (tavily) if:
- The question requires current/real-time information
- The document database doesn't contain relevant information

You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that! """

citations = []

@st.cache_resource


def get_chatbot():
    return Chatbot(system=system)
logging.basicConfig(
                level="INFO",
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True)]
            )



os.makedirs("temp", exist_ok=True)
chatbot = get_chatbot()


def get_filepaths(uploaded_files):
    filepaths = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with open(f"temp/temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        filepaths.append(f"temp/temp_{uploaded_file.name}")
 
    return filepaths

def smart_highlight(page, search_text):
    """
    Attempts to highlight text on a page using two strategies:
    1. Exact Match: Searches for the exact string.
    2. Anchor Match: Searches for the first 5 words (Head) and last 5 words (Tail).
    """
    # Strategy 1: Exact Match (Fastest)
    # quads=True allows efficient highlighting of slanted/complex text
    text_instances = page.search_for(search_text, quads=True)
    
    if text_instances:
        for inst in text_instances:
            page.add_highlight_annot(inst)
        return True

    # Strategy 2: Anchor Match (Robust to newlines/hyphens)
    words = search_text.split()
    
    # Only try anchor search if the text is long enough (e.g., > 10 words)
    if len(words) < 10:
        return False
        
    # Define Anchors (First 5 words and Last 5 words)
    head_text = " ".join(words[:5])
    tail_text = " ".join(words[-5:])
    
    # Search for instances of head and tail
    head_instances = page.search_for(head_text, quads=True)
    tail_instances = page.search_for(tail_text, quads=True)
    
    if head_instances and tail_instances:
        # We found both start and end phrases. 
        # Logic: Find a tail that appears AFTER a head.
        
        # Take the first valid pair we find (Simple greedy approach)
        # In a production app, you might compare Y-coordinates to ensure they are close.
        valid_match = False
        
        for head in head_instances:
            for tail in tail_instances:
                # Check if tail is visually "after" head (simplistic check using rect objects)
                # PyMuPDF Quads/Rects: (x0, y0, x1, y1)
                # We check if tail.y1 (bottom) >= head.y0 (top)
                if tail.lr.y >= head.ul.y: 
                    # Highlight the Start (Greenish)
                    annot_head = page.add_highlight_annot(head)
                    annot_head.set_colors(stroke=(0.5, 1, 0.5)) # Light Green
                    annot_head.update()
                    
                    # Highlight the End (Reddish)
                    annot_tail = page.add_highlight_annot(tail)
                    annot_tail.set_colors(stroke=(1, 0.5, 0.5)) # Light Red
                    annot_tail.update()
                    
                    # Optional: Draw a line connecting them to show the relationship
                    # p1 = head.lr  # Bottom-right of head
                    # p2 = tail.ul  # Top-left of tail
                    # page.draw_line(p1, p2, color=(1, 1, 0), width=2)
                    
                    valid_match = True
        
        return valid_match

    return False

def find_and_display_page(pdf_file, search_text, target_page_num=None):
    # 1. Open the PDF
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    found_page_index = -1
    
    # OPTIMIZATION: If we already know the page number from RAG, check that first!
    if target_page_num is not None:
        # Adjust for 0-based index vs 1-based user input
        target_idx = int(target_page_num) - 1 
        if 0 <= target_idx < len(doc):
            page = doc[target_idx]
            if smart_highlight(page, search_text):
                found_page_index = target_idx
    
    # If not found yet (or no page hint provided), Scan ALL pages
    if found_page_index == -1:
        for i, page in enumerate(doc):
            if smart_highlight(page, search_text):
                found_page_index = i
                break
            
    if found_page_index != -1:
        st.success(f"Found text on page {found_page_index + 1}")
        
        # 4. Render the specific page to an image
        page = doc.load_page(found_page_index)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return img
    else:
        st.error(f"Text could not be matched exactly. Try a shorter snippet.")
        return None

def display_citations(citations):

    with st.expander("Show citations"):
        for citation in citations:
            st.info(f"Referenced from {citation['source']} (Page {citation['page']})")
        try: 
            if citation.get("image"):
                st.image(citation["image"], use_container_width=True)
            else:
                st.warning("Citation image could not be generated")
        except Exception as e:
            st.error(f"Error displaying citations: {str(e)}")
            logging.error(f"Citation display error: {e}")
    
# Page configuration
st.set_page_config(page_title="Chatbot Interface", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []


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
# st.title(st.session_state.current_page)

if not st.session_state.messages:

    default_html = '<p style="color: grey; text-align: center; margin-top: 200px;">Start a conversation...</p>'
    st.markdown(default_html, unsafe_allow_html=True)
for message in st.session_state.messages:
    logging.info(f"Rendering message: {st.session_state.messages}")
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message["role"] == "assistant":
            st.markdown(message["content"]["response"])
            citations = message["content"].get("citations", [])

            display_citations(citations)
            

if prompt := st.chat_input("Type your message here...", key="input_text"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt}) 

        bot_response = chatbot.query_chatbot(prompt)
        if len(bot_response.get("citations", [])) > 0:
            # Show citations
            for i, cite in enumerate(bot_response["citations"]):
                source = cite.get("source")
                page = cite.get("page")
                cited_content = cite.get("content")
                citation_image = find_and_display_page(
                    open(source, "rb"), 
                    cited_content,
                    target_page_num=page)
                if citation_image:
                    bot_response["citations"][i]["image"] = citation_image
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        st.rerun()

            

        
        


