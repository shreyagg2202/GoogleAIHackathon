# #Imports
# # __import__('pysqlite3')
# # import sys
# # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import chromadb
# import streamlit as st
# import google.generativeai as genai
# import time
# import random
# from chromadb import Documents, EmbeddingFunction, Embeddings

# #initialization
# API_KEY = "AIzaSyD69RPGzZPIDHTRzIWQH887huukyD5cHHc"

# st.set_page_config(
#     page_title="Policy Bazaar",
#     page_icon="🔥"
# )

# # chroma_client = chromadb.PersistentClient(path="Chroma_DB/")


# st.title("Policy Bazaar")
# st.caption("A policy advisor powered by Google Gemini")

# # if "app_key" not in st.session_state:
# #     app_key = st.text_input("Please enter your Gemini API Key", type='password')
# #     if app_key:
# #         st.session_state.app_key = app_key

# if "history" not in st.session_state:
#     st.session_state.history = []

# try:
#     genai.configure(api_key = API_KEY)
# except AttributeError as e:
#     st.warning("API Key not working")

# # class GeminiEmbeddingFunction(EmbeddingFunction):
# #   def __call__(self, input: Documents) -> Embeddings:
# #     model = 'models/text-embedding-004'
# #     return genai.embed_content(model=model,
# #                                 content=input,
# #                                 task_type="question_answering")["embedding"]
  
# # embedding_function = GeminiEmbeddingFunction()

# # db = chroma_client.get_or_create_collection(name="Test3", embedding_function=embedding_function)

# # def get_relevant_passage(query_embedding, db):
# #   passage = db.query(query_embeddings=[query_embedding], n_results=3)['documents'][0]
# #   return passage

# # def make_prompt(query):

# #     query_embeddings = embedding_function(query)
# #     relevant_passage = get_relevant_passage(query_embeddings[0], db)
# #     relevant_passage = "\n\n---\n\n".join(relevant_passage)
# #     #   escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
# #     prompt = ("""INSTRUCTIONS:
# #     Answer the users QUESTION using the DOCUMENT text above.
# #     Keep your answer ground in the facts of the DOCUMENT.
# #     If the DOCUMENT doesn't contain the facts to answer the QUESTION return "I dont have sufficient information for this query"

# #     DOCUMENT:
# #     {relevant_passage}

# #     QUESTION:
# #     {query}

# #     """).format(query=query, relevant_passage=relevant_passage)

# #     return prompt


# generation_config = {
#     "temperature": 0.2,
#     "top_p": 0.95,
#     "top_k": 64,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     system_instruction="""You are an Insurance Assistant chatbot tasked with helping users understand their insurance policies, 
#     answer coverage questions, handle claims, and simplify complex insurance terminology. Your responses should be clear, 
#     accurate, and user-friendly. If specific policy details are unknown, instruct the user to contact customer support for 
#     precise information. Maintain a professional tone, avoid legal advice, and request additional information if needed to 
#     accurately address the user's query. Your goal is to make the insurance process understandable and ensure users feel 
#     confident about their insurance decisions.""",
#     generation_config=generation_config,
# )

# # model = genai.GenerativeModel("gemini-pro")
# chat = model.start_chat(history = st.session_state.history)

# with st.sidebar:
#     if st.button("Clear Chat Window", use_container_width=True, type="primary"):
#         st.session_state.history = []
#         st.rerun()

# for message in chat.history:
#     role ="assistant" if message.role == 'model' else message.role
#     with st.chat_message(role):
#         st.markdown(message.parts[0].text)

# i=1
# while True:
#     if prompt := st.chat_input("Enter your input here:", key=f"input{i}"):
#         i+=1
#         prompt = prompt.replace('\n', ' \n')
#         # prompt = make_prompt(prompt)
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             message_placeholder.markdown("Thinking...")
#             try:
#                 full_response = ""
#                 for chunk in chat.send_message(prompt, stream=True):
#                     word_count = 0
#                     random_int = random.randint(5,10)
#                     for word in chunk.text:
#                         full_response+=word
#                         word_count+=1
#                         if word_count == random_int:
#                             time.sleep(0.05)
#                             message_placeholder.markdown(full_response + "_")
#                             word_count = 0
#                             random_int = random.randint(5,10)
#                 message_placeholder.markdown(full_response)
#             except genai.types.generation_types.BlockedPromptException as e:
#                 st.exception(e)
#             except Exception as e:
#                 st.exception(e)
#             st.session_state.history = chat.history


# Imports
import streamlit as st
import google.generativeai as genai
import time
import random

# Initialization
API_KEY = "AIzaSyD69RPGzZPIDHTRzIWQH887huukyD5cHHc"

st.set_page_config(page_title="Policy Bazaar", page_icon="🔥")
st.title("Policy Bazaar")
st.caption("A policy advisor powered by Google Gemini")

# Configure the Gemini API
genai.configure(api_key=API_KEY)

# Define the model configuration
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
chat = model.start_chat(history = st.session_state.history)

with st.sidebar:
    if st.button("Clear Chat Window", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.rerun()

for message in chat.history:
    role ="assistant" if message.role == 'model' else message.role
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

while True:
    if prompt := st.chat_input(""):
        prompt = prompt.replace('\n', ' \n')
        prompt = make_prompt(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                full_response = ""
                for chunk in chat.send_message(prompt, stream=True):
                    word_count = 0
                    random_int = random.randint(5,10)
                    for word in chunk.text:
                        full_response+=word
                        word_count+=1
                        if word_count == random_int:
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "_")
                            word_count = 0
                            random_int = random.randint(5,10)
                message_placeholder.markdown(full_response)
            except genai.types.generation_types.BlockedPromptException as e:
                st.exception(e)
            except Exception as e:
                st.exception(e)
            st.session_state.history = chat.history
