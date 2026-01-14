from langchain_core.tools import tool
import os
from pyexpat import model
from unittest import result
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain_contextual import ContextualRerank
from transformers import AutoTokenizer



class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]

class Chatbot:
    

    def create_retrieve_documents_tool(self):
        """Create a tool version of retrieve_documents"""
        @tool
        def retrieve_documents(query: str):
            """Search and return information relevant to the query, within the stored database"""
            docs = self.retriever.invoke(query)
            reranked_documents = self.compressor.compress_documents(
                query=query,
                documents=docs,
                top_n=10
            )
            return "\n\n".join([doc.page_content for doc in reranked_documents])
        
        return retrieve_documents

    def __init__(self, system=None):



        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
        os.environ["TAVILY_API_KEY"] = os.getenv('TavilyAPI')
        os.environ["CONTEXTUAL_AI_API_KEY"] = os.getenv('reranker')


        self.system = system        

        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20, length_function=self.huggingface_token_len)
        self.persist_directory = "./chroma_langchain_db"
        self.store = {}
        self.vector_store = None
        self.initialize_vector_store()
        self.retriever = self.vector_store.as_retriever(search_type = 'mmr') if self.vector_store else None
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        search_tool = TavilySearch(max_results=5)
        self.checkpointer = InMemorySaver()
        self.reranker_model = "ctxl-rerank-v2-instruct-multilingual"
        self.compressor = ContextualRerank(
            model=self.reranker_model,
            api_key=os.environ["CONTEXTUAL_AI_API_KEY"])
        self.tools = [self.create_retrieve_documents_tool(), search_tool]
        self.graph = self.create_graph(self.tools)
        self.thread = '1' 

    def huggingface_token_len(self, text):
        return len(self.tokenizer.encode(text))

    def initialize_vector_store(self):
        self.vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def load_embeddings(self, filepaths):
        docs = []

        # Define the directory
        # Iterate over everything in the directory
        for document in filepaths:
            print(f"loading document: {document}")

            if Path(document).suffix == '.pdf':
                docs.extend(PyPDFLoader(document).load())
            elif Path(document).suffix == '.docx':
                docs.extend(UnstructuredWordDocumentLoader(document).load())

            print(f"Loaded {document}")


        
        splits = self.text_splitter.split_documents(docs)
        len(splits)

        batch_size = 4000
        for i in range(0, len(splits), batch_size):
            self.vector_store.add_documents(splits[i : i + batch_size])

        
        print(self.vector_store._collection.count())
    

    def create_graph(self, tools):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.call_action )
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        graph = graph.compile(checkpointer = self.checkpointer)
        # tools_list = {t.name: t for t in tools}
        # print("PRINTING LIST OF TOOLS:")
        # print(tools_list)
        self.model = self.llm.bind_tools(tools)
        return graph

    
    def delete_document(self, filename):
        # 1. Reconstruct the path used during ingestion (Must match exactly)
        # Based on your previous code: f"./temp_{uploaded_file.name}"
        temp_path = f"./temp_{filename}"
        
        print(f"Removing vectors for source: {temp_path}")

        # 2. Find IDs of all chunks where metadata 'source' matches this path
        # Chroma allows filtering by metadata using the 'where' clause
        results = self.vector_store.get(where={"source": temp_path})
        
        ids_to_delete = results.get('ids')

        # 3. Delete them
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            print(f"Successfully deleted {len(ids_to_delete)} chunks.")
            return True
        else:
            print("No chunks found for this file.")
            return False
        
    def get_existing_documents(self):
        """
        Retrieves all unique source filenames currently in the vector store.
        Returns: A set of filenames (e.g., {'report.pdf', 'invoice.docx'})
        """
        try:
            # Get all metadata from the collection
            # We only need the 'source' field to identify files
            data = self.vector_store.get(include=['metadatas'])
            
            unique_sources = set()
            for metadata in data['metadatas']:
                if metadata and 'source' in metadata:
                    source_path = metadata['source']
                    # Clean the path to get just the filename
                    # Assumes path is like "./temp_filename.pdf"
                    filename = os.path.basename(source_path).replace("temp_", "", 1)
                    unique_sources.add(filename)
            
            return unique_sources
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return set()

    def query_chatbot(self, query: str):
         messages = [HumanMessage(content=query)]
         response = self.graph.invoke({"messages": messages}, config={
             "configurable": {
                 "thread_id": '1'}
                 })    
         
         return parse_output(response['messages'][-1].content)


    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0


    # NODES

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def call_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        tools_by_name = {tool.name: tool for tool in self.tools}

        print(f"Available tools: {list(tools_by_name.keys())}")  # Debug: see what tools are available

        results = []
        for t in tool_calls:
            print(f"Calling {t}")
            if  (t['name']) not in tools_by_name:
                print(f"Bad tool call - '{t['name']}' not in {list(tools_by_name.keys())}")
                result = 'bad tool name, retry'
            else:
                result = tools_by_name[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    

    # TOOLS

    

def format_grouped_sources(docs):
    # Dictionary to hold source -> set of pages
    sources_data = defaultdict(set)

    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Source')
        # Use 'page_label' or fallback to 'page'
        page = str(doc.metadata.get('page_label', doc.metadata.get('page', 'N/A')))
        sources_data[source].add(page)

    # Create the formatted string
    formatted_sources = []
    for source, pages in sources_data.items():
        # Sort pages for cleaner look
        sorted_pages = sorted(list(pages))
        pages_str = ", ".join(sorted_pages)
        formatted_sources.append(f"Source: {source}, Pages: {pages_str}")
        
    return "; ".join(formatted_sources)


def parse_output(content):
        """Extract text from various content formats"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle list of content blocks with metadata
            texts = []
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    texts.append(block['text'])
            return ''.join(texts) if texts else str(content)
        else:
            return str(content)