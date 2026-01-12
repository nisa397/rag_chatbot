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



class Chatbot:


    def __init__(self):
        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.persist_directory = "./chroma_langchain_db"
        self.store = {}
        self.vector_store = None
        self.initialize_vector_store()
        self.retriever = self.vector_store.as_retriever(search_type = 'mmr') if self.vector_store else None
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. At the end of your answer, explicitly list the sources exactly as provided in the "Sources" section below. Keep the answer as concise as possible.  """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # <--- Memory goes here
            ("user", "{question}"),
            ("system", "Context: {context}, Sources: {sources}")
        ])
        self.chain = (
            RunnablePassthrough.assign(
                context=lambda x: self.retriever.invoke(x["question"]),
            )
            .assign(
                sources=lambda x: format_grouped_sources(x['context'])
            )
            .assign(answer= self.prompt | self.llm)
        )
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer"# Must match the prompt placeholder
        )


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

    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def query_chatbot(self, question: str):
        response = self.chain_with_history.invoke(
            {"question": question},
            config={
                "configurable": {
                    "session_id": "user_session_1"
                    }
                    })
        return response['answer'].content
    
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


