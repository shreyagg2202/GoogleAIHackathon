#Imports
__import__('pysqlite3')
import sys
import os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import streamlit as st
import google.generativeai as genai
import joblib
import time
from chromadb import Documents, EmbeddingFunction, Embeddings

#initialization
API_KEY = "AIzaSyD69RPGzZPIDHTRzIWQH887huukyD5cHHc"

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

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
# st.write(chroma_client.list_collections())
# st.write(db.get(include=['metadatas','documents']))

def get_relevant_passage(query_embedding, db):
#   st.write("db:" ,db.get())
  passage = db.query(query_embeddings=[query_embedding], n_results=3)['documents'][0]
  return passage

def make_prompt(query):

    query_embeddings = embedding_function(query)
    # st.write("query_embeddings:" ,query_embeddings)
    relevant_passage = get_relevant_passage(query_embeddings[0], db)
    # st.write("relevant_passage:" ,relevant_passage)
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
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""You are a human that provides Insurance Assistance, tasked with helping users understand their insurance policies, 
    answer coverage questions and simplify complex insurance terminology. You will refer to the provided context and then base your 
    answers on that without ever mentioning your backend documents in an upfront manner. Your responses should be clear, 
    accurate, and user-friendly. If specific policy details are unknown, instruct the user to contact customer support for 
    precise information. Maintain a professional tone, avoid legal advice, and request additional information if needed to 
    accurately address the user's query. Your goal is to make the insurance process understandable and ensure users feel 
    confident about their insurance decisions.
    
    Limitations: If the user engages in normal conversation, then you should reply appropriately""",
    generation_config=generation_config,
)

# def translate_role_for_streamlit(user_role):
#     if user_role == "model":
#         return "assistant"
#     else:
#         return user_role
    
# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Past Chats')
    chat_id = st.text_input('Create or select a chat ID', value=time.strftime('%Y%m%d%H%M%S'))
    chat_name = st.text_input('Name your chat', value=past_chats.get(chat_id, ''))
    if st.button('Save Chat'):
        past_chats[chat_id] = chat_name
        joblib.dump(past_chats, 'data/past_chats_list')

    chat_selection = st.selectbox('Select a chat session', options=list(past_chats.keys()), format_func=lambda x: past_chats[x])
    st.session_state.chat_id = chat_selection if chat_selection else chat_id
    st.session_state.chat_title = past_chats.get(st.session_state.chat_id, 'New Chat')

     # Option to delete selected chat
    if st.button('Delete Selected Chat'):
        if st.session_state.chat_id in past_chats:
            # Delete chat records
            os.remove(f'data/{st.session_state.chat_id}-st_messages')
            del past_chats[st.session_state.chat_id]
            joblib.dump(past_chats, 'data/past_chats_list')
            st.experimental_rerun()
            
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []
    print('new_cache made')
    st.session_state.model = model
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history,
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])


# React to user input
if prompt := st.chat_input('Your message here...'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    ## Send message to AI
    user_prompt = make_prompt(prompt)
    response = st.session_state.chat.send_message(
        user_prompt,
        stream=True,
    )
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        assistant_response = response
        # Streams in a chunk at a time
        for chunk in response:
            # Simulate stream of chunk
            # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
            for ch in chunk.text.split(' '):
                full_response += ch + ' '
                time.sleep(0.05)
                # Rewrites with a cursor at end
                message_placeholder.write(full_response + '▌')
        # Write full message with placeholder
        message_placeholder.write(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )

# # Initialize chat session in Streamlit if not already present
# if "chat_session" not in st.session_state:
#     st.session_state.chat_session = model.start_chat(history=[])

# # Display the chat history
# for message in st.session_state.chat_session.history:
#     with st.chat_message(translate_role_for_streamlit(message.role)):
#         st.markdown(message.parts[0].text)

# # Input field for user's message
# user_prompt = st.chat_input("Ask Gemini-Pro...")
# if user_prompt:
#     # Add user's message to chat and display it
#     st.chat_message("user").markdown(user_prompt)

#     # Send user's message to Gemini-Pro and get the response
#     prompt = make_prompt(user_prompt)
#     gemini_response = st.session_state.chat_session.send_message(prompt)

#     # Display Gemini-Pro's response
#     with st.chat_message("assistant"):
#         st.markdown(gemini_response.text)