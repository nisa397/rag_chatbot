from chatbot import Chatbot
import logging
from rich.logging import RichHandler

logging.basicConfig(
                level="INFO",
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True)]
            )


system= """You are a helpful assistant with the main purpose of answering queries related to the documents you are given. You have access to a document database and web search to use if the documents don't contain the information needed.

IMPORTANT: Always try to answer questions using the retrieve_documents tool FIRST, 
as it searches your internal knowledge base. Only use web search (tavily) if:
- The question requires current/real-time information
- The document database doesn't contain relevant information

You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that! """
chatbot = Chatbot(system=system)

logging.info("INITIALIZED CHATBOT")
chatbot.load_embeddings(r"C:\Users\mailm\Documents\GitHub\rag_chatbot\chatbot\temp\temp_Apple_10-K-2021.pdf")

logging.info("LOADING EMBEDDINGS AND VECTORSTORE...")
logging.info(chatbot.vector_store)

logging.info(chatbot.vector_store._collection.count())

query = "What is Apple?"

logging.info(f"ASKING QUERY: {query}")

response = chatbot.query_chatbot(query)

logging.info("RESPONSE:")
logging.info(response)

