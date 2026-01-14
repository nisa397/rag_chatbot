from chatbot import Chatbot

system= "You are a smart research assistant. Utilize the given documents in the database to retrieve information via the retrieve_documents tool. If you are unable to find any relevant information, you can utilize the TavilySearch tool. You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that! "
chatbot = Chatbot(system=system)

# print("TOOLS LIST:")
# print("PRINTING TYPE OF TOOLS LIST:")
# print(type())
# print("PRINTING LENGTH OF TOOLS LIST:")
# print(len(chatbot.tools_list))
# print(chatbot.tools_list)

print("LOADING EMBEDDINGS AND VECTORSTORE...")
print(chatbot.vector_store)
print(chatbot.vector_store._collection.count())

query = "What is Tesla, and who are their competitors?"

print(f"ASKING QUERY: {query}")

response = chatbot.query_chatbot(query)

print("RESPONSE:")
print(response)

