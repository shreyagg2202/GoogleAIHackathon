#Imports
__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import streamlit as st
import google.generativeai as genai
import time
import random
from chromadb import Documents, EmbeddingFunction, Embeddings

#initialization
API_KEY = "AIzaSyD69RPGzZPIDHTRzIWQH887huukyD5cHHc"

st.set_page_config(
    page_title="Policy Bazaar",
    page_icon="🔥"
)

chroma_client = chromadb.PersistentClient(path="Chroma_DB/")

# files_in_chroma_db = os.listdir('Chroma_DB')
# st.write("Files in Chroma_DB:", files_in_chroma_db)

st.title("Policy Bazaar")
st.caption("A policy advisor powered by Google Gemini")

# if "app_key" not in st.session_state:
#     app_key = st.text_input("Please enter your Gemini API Key", type='password')
#     if app_key:
#         st.session_state.app_key = app_key

if "history" not in st.session_state:
    st.session_state.history = []

try:
    genai.configure(api_key = API_KEY)
except AttributeError as e:
    st.warning("API Key not working")

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/text-embedding-004'
    return genai.embed_content(model=model,
                                content=input,
                                task_type="question_answering")["embedding"]
  
embedding_function = GeminiEmbeddingFunction()

db = chroma_client.get_collection(name="Test3", embedding_function=embedding_function)
st.write(chroma_client.list_collections())
st.write(db.get())

def get_relevant_passage(query_embedding, db):
  st.write("db:" ,db.get())
  passage = db.query(query_embeddings=[query_embedding], n_results=3)['documents'][0]
  return passage

def make_prompt(query):

    query_embeddings = embedding_function(query)
    st.write("query_embeddings:" ,query_embeddings)
    relevant_passage = get_relevant_passage(query_embeddings[0], db)
    st.write("relevant_passage:" ,relevant_passage)
    relevant_passage = "\n\n---\n\n".join(relevant_passage)
    #   escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""INSTRUCTIONS:
    Answer the users QUESTION using the DOCUMENT text above.
    Keep your answer ground in the facts of the DOCUMENT.
    If the DOCUMENT doesn't contain the facts to answer the QUESTION return "I dont have sufficient information for this query"

    DOCUMENT:
    {relevant_passage}

    QUESTION:
    {query}

    """).format(query=query, relevant_passage=relevant_passage)

    return prompt


generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""You are an Insurance Assistant chatbot tasked with helping users understand their insurance policies, 
    answer coverage questions, handle claims, and simplify complex insurance terminology. Your responses should be clear, 
    accurate, and user-friendly. If specific policy details are unknown, instruct the user to contact customer support for 
    precise information. Maintain a professional tone, avoid legal advice, and request additional information if needed to 
    accurately address the user's query. Your goal is to make the insurance process understandable and ensure users feel 
    confident about their insurance decisions.""",
    generation_config=generation_config,
)

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
    
# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Display the chat history
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Ask Gemini-Pro...")
if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)

    # Send user's message to Gemini-Pro and get the response
    prompt = make_prompt(user_prompt)
    gemini_response = st.session_state.chat_session.send_message(prompt)

    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)