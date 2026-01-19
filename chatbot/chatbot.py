from urllib import response
from langchain_core.tools import tool
import os
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
from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain_contextual import ContextualRerank
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from typing import Optional
import logging
from rich.logging import RichHandler


class SearchParams(BaseModel):
    query: str = Field(..., description="The semantic search query")
    document_titles: Optional[List[str]] = Field(None, description="The specific document title or titles to search in, if mentioned")


class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        citations: Annotated[list[dict], lambda x, y: y]

class Chatbot:

    def __init__(self, system=None):

        # Create and configure logging
        logging.basicConfig(
                level="INFO",
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True)]
            )
                    
        



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
        logging.info("Chatbot initialized with tools and graph.")

    def huggingface_token_len(self, text):
        return len(self.tokenizer.encode(text))

    def initialize_vector_store(self):
        self.vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        logging.info("Vector store initialized.")

    def load_embeddings(self, filepaths):
        docs = []

        # Define the directory
        # Iterate over everything in the directory
        for document in filepaths:
            print(f"loading document: {document}")
            logging.info(f"Loading document: {document}")

            if Path(document).suffix == '.pdf':
                docs.extend(PyPDFLoader(document).load())
            elif Path(document).suffix == '.docx':
                docs.extend(UnstructuredWordDocumentLoader(document).load())

            print(f"Loaded {document}")
            logging.info(f"Loaded document: {document}")


        
        splits = self.text_splitter.split_documents(docs)
        len(splits)

        batch_size = 4000
        for i in range(0, len(splits), batch_size):
            self.vector_store.add_documents(splits[i : i + batch_size])

        
        print(self.vector_store._collection.count())
        logging.info(f"Total documents in vector store: {self.vector_store._collection.count()}")
        logging.info("Embeddings loaded successfully.")
    
    def create_graph(self, tools):
        logging.info("Creating state graph for the agent.")
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.call_action )
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        graph = graph.compile(checkpointer = self.checkpointer)
        logging.info(f"tools list: {tools}")
        self.model = self.llm.bind_tools(tools)
        logging.info("State graph created successfully.")


        return graph
    
    def delete_document(self, filename):
        # 1. Reconstruct the path used during ingestion (Must match exactly)
        # Based on your previous code: f"./temp_{uploaded_file.name}"
        temp_path = f"temp/temp_{filename}"
        
        logging.info(f"Removing vectors for source: {temp_path}")

        # 2. Find IDs of all chunks where metadata 'source' matches this path
        # Chroma allows filtering by metadata using the 'where' clause
        results = self.vector_store.get(where={"source": temp_path})
        
        ids_to_delete = results.get('ids')

        # 3. Delete them
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            logging.info(f"Successfully deleted {len(ids_to_delete)} chunks.")
            return True
        else:
            logging.info("No chunks found for this file.")
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
            logging.error(f"Error fetching documents: {e}")
            return set()

    def query_chatbot(self, query: str):
         messages = [HumanMessage(content=query)]
         response = self.graph.invoke({"messages": messages}, config={
             "configurable": {
                 "thread_id": '1'}
                 })
         
         final_text = parse_output(response['messages'][-1].content)
         
         final_citations = response.get('citations', [])
         return {
             "response": final_text,
             "citations": final_citations
            }

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

        logging.info(f"Available tools: {list(tools_by_name.keys())}")  # Debug: see what tools are available

        results = []
        new_citations = []
        for t in tool_calls:
            logging.info(f"Calling {t}")
            if  (t['name']) not in tools_by_name:
                logging.error(f"Bad tool call - '{t['name']}' not in {list(tools_by_name.keys())}")
                result = 'bad tool name, retry'
            else:
                tool_output = tools_by_name[t['name']].invoke(t['args'])
                logging.info("Tool returned result.")
                logging.info(tool_output)
                # CHECK: Is this our document tool returning rich data?
                if isinstance(tool_output, dict) and "citations" in tool_output:
                    # 1. Give the LLM ONLY the text content
                    result = tool_output["context_text"]
                    # 2. Save the citations to our state list
                    new_citations = tool_output["citations"]
                else:
                    result = tool_output
                

                


        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logging.info("Back to the model!")
        return {'messages': results, 'citations': new_citations}
    

    # TOOLS
    def create_retrieve_documents_tool(self):
        """Create a tool version of retrieve_documents"""
        @tool
        def retrieve_documents(query: str):
            """Search and return information relevant to the query, within the stored database. """

            structured_llm = self.llm.with_structured_output(SearchParams)

            available_docs = self.get_existing_documents()
    
            if not available_docs:
                return None
            
            docs_list = ", ".join(available_docs)
            prompt = f"""Given this query: "{query}"
            and these
            Available documents: {docs_list}

            Should this query search in specific documents? Respond with just the document name from the given list, if yes, otherwise respond with an empty list."""

            # 3. Run the extraction
            params = structured_llm.invoke(prompt)
            # Result: params.query='holidays', params.document_title='HR Manual'

            

            if params.document_titles:
                logging.info(f"Routing search to documents: {params.document_titles}")
                search_kwargs = {"k": 20}
                # Construct file paths (adjust formatting as per your storage logic)
                target_sources = [f"temp/temp_{title}" for title in params.document_titles]
                
                # KEY FIX: Handle single vs multiple documents for vector store filtering
                # Most vector stores (Chroma, Pinecone) support an "$in" operator for lists
                if len(target_sources) == 1:
                    search_kwargs["filter"] = {"source": target_sources[0]}
                else:
                    search_kwargs["filter"] = {"source": {"$in": target_sources}}
                
                # Use the ORIGINAL 'query' here, not params.query (unless you add query rewriting)
                docs = self.vector_store.max_marginal_relevance_search(
                    query, 
                    **search_kwargs
                )
            else:
                logging.info("Searching across all documents.")
                docs = self.retriever.invoke(query)
            
            if not docs:
                return "No relevant documents found."

            reranked_documents = self.compressor.compress_documents(
                query=query,
                documents=docs,
                top_n=5
            )

            logging.info(f"Reranked documents:")
            logging.info(reranked_documents)
            # Collect reranked documents metadata: Source, page number and content
            reranked_documents = [doc for doc in reranked_documents if doc.metadata['relevance_score'] > 0.8]
            citations = format_grouped_sources(reranked_documents)
            logging.info("Citations for reranked documents:")
            context_text = "\n\n".join([doc.page_content for doc in reranked_documents])

            return {
                "context_text": context_text,
                "citations": citations
            }
        
        return retrieve_documents
    

def format_grouped_sources(docs):
    # Dictionary to hold source -> set of pages
    citations = []

    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Source')
        # Use 'page_label' or fallback to 'page'
        page = str(doc.metadata.get('page_label', doc.metadata.get('page', 'N/A')))

        content =  clean_text(doc.page_content)
        citations.append({'source': source, 'page': page, 'content': content})
        
    return citations


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
        
def clean_text(text):
    """Convert escape sequences to spaces and normalize whitespace"""
    # Replace escape sequences with spaces
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    # Remove extra/multiple spaces
    text = ' '.join(text.split())
    return text



