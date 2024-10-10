#Imports
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import streamlit as st
import google.generativeai as genai

import time
from chromadb import Documents, EmbeddingFunction, Embeddings
from semantic_search import is_similar

from typing import Optional

from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI   # Import the LLM

#initialization
API_KEY = "AIzaSyD69RPGzZPIDHTRzIWQH887huukyD5cHHc"

# new_chat_id = time.strftime('%Y%m%d%H%M%S')
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

st.set_page_config(
    page_title="Policy Bazaar",
    page_icon="🔥"
)

chroma_client = chromadb.PersistentClient(path="Chroma_DB/")


st.title("Policy Bazaar")
st.caption("A policy advisor powered by Google Gemini")

try:
    genai.configure(api_key=API_KEY)
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
    prompt = (f"""
            INSTRUCTIONS:
            You are an expert on insurance policies, and you are having a conversation where you help the user with their questions. 
            Use your extensive knowledge to provide clear, concise, and accurate answers without referring to any documents directly. 
            If the information needed to answer the question fully is not available, advise the user to contact customer support for 
            more specific details.

            QUESTION:
            {query}

            CONTEXT:
            As an expert, you know the following about similar topics:
            {relevant_passage}

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
    system_instruction="""
        You are a human-like insurance expert tasked with helping users understand their insurance policies, 
        answer coverage questions, and simplify complex insurance terminology. Your responses should rely on 
        your vast knowledge of insurance matters, which you will convey clearly and accurately without explicitly 
        referring to the documents you have access to. You should appear knowledgeable and only suggest contacting 
        customer support or providing general guidance when the details are not within your immediate knowledge scope. 
        Maintain a professional tone, avoid giving legal advice, and ask for more details if necessary to provide precise answers. 
        Your goal is to make the insurance process understandable and ensure users feel confident and well-informed about their decisions.

        Limitations: If the user engages in casual conversation or asks about general knowledge, respond in a manner that maintains 
        the facade of a human expert. You should not mention or imply that your responses are generated or sourced from specific documents.
        """,
    generation_config=generation_config,
)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []

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

    if is_similar(prompt):
        st.write("Are you sure you want to select this policy?")

        # Use Streamlit buttons for user confirmation
        confirm = st.button("Yes, proceed")
        cancel = st.button("No, continue chatting")

        if confirm:
            st.write("Thanks for selecting a policy and trusting us. I will now ask you a set of questions to gather your details for further processing of the selected policy.")
            # Set a flag in session state to indicate policy selection
            st.session_state.policy_selected = True
            # Proceed to the next step (e.g., collecting user details)
            # You can implement the next steps here
        elif cancel:
            st.write("Okay, let's continue our conversation.")
            st.session_state.policy_selected = False
        else:
            # Wait for the user's response
            st.stop()

    if not st.session_state.get('policy_selected', False):
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
    else:
        # Proceed to the next step since the policy has been selected
        # Implement your code for collecting user details here
        st.write("Proceeding to collect user details...")
        # You can create a form or inputs to collect user information
        name = st.text_input("Enter your full name")
        email = st.text_input("Enter your email address")
        # Add more fields as necessary
        
        # You can process or save this information as needed
        if st.button("Submit Details"):
            st.write("Thank you! Your details have been submitted.")
            # Reset the conversation or take further action

# API_key = "AIzaSyD69RPGzZPIDHTRzIWQH887huukyD5cHHc"

class PersonalDetails(BaseModel):
    name: Optional[str] = Field(default="",
        description = "This is the name of the user.",
    )
    date_of_birth: Optional[str] = Field(default="",
        description = "This is the date of birth of the user.",
    )
    address: Optional[str] = Field(default="",
        description = "This is the address of the user.",
    )
    phone_number: Optional[str] = Field(default="",
        description = "This is the phone_number of the user.",
    )
    email_address: Optional[str] = Field(default="",
        description = "This is the email address of the user.",
    )

def check_what_is_empty(user_personal_details):
    ask_for=[]
    # Check if fields are empty
    for field, value in user_personal_details.dict().items():
        if value in [None, "", 0]:
            # print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for


# Checking the response and adding it
def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails):
    if new_details != None:
        non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details
    return current_details

def filter_response(text_input, user_details):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_key)
    chain = llm.with_structured_output(PersonalDetails)
    res = chain.invoke(text_input)
    user_details = add_non_empty_details(user_details,res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for

# Main function to conduct the conversation
def collect_user_details(ask_for=None):
    if ask_for is None:
        ask_for = ['name', 'date_of_birth', 'address', 'phone_number', 'email_address', 'adhaar_number']

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_key)

    # Initialize the conversation history
    messages = []
    
    user_personal_details = PersonalDetails(full_name="",
                                            date_of_birth="",
                                            address="",
                                            phone_number="",
                                            email="",
                                            adhaar_number="")

    while ask_for:
        # Prepare the list of remaining items
        remaining_items = ', '.join(ask_for)

        # Define the system message with the remaining items
        system_message_content = f"""
You are an assistant that needs to collect the following information from the user: {remaining_items}.
- Ask for one item at a time in a conversational manner.
- Do not mention items that have already been provided.
- If there are no more items left, thank the user and inform them that the Customer Support team will contact them soon.
"""
        # Create the prompt template
        prompt = ChatPromptTemplate(
            [
                SystemMessage(content=system_message_content),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        # Build the chain by combining the prompt and the LLM
        chain = prompt | llm

        time.sleep(5)
        ai_message = chain.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Please go ahead and ask the required questions."
                        ),
                ],
            }
        )
        
        # print(messages)
        print(f"Assistant: {ai_message.content}")

        # Append the assistant's message to the conversation history
        messages.append(AIMessage(content=ai_message.content))
            
        # Get user input
        user_input = input("User: ")

        # Check if the user wants to restart
        if user_input.strip().lower() == 'restart':
            # Re-initialize user_personal_details and ask_for
            user_personal_details = PersonalDetails(
                full_name="",
                date_of_birth="",
                address="",
                phone_number="",
                email="",
                adhaar_number=""
            )
            ask_for = ['name', 'date_of_birth', 'address', 'phone_number', 'email_address', 'adhaar_number']
            print("Assistant: Let's start over. Please provide your details again.")
            # Optionally, clear the conversation history
            messages = []
            continue

        # Append the user's message to the conversation history
        messages.append(HumanMessage(content=user_input))
        
        user_details, ask_for = filter_response(user_input, user_personal_details)
        user_personal_details = user_details
        
    
    # print(user_details)
    return user_details
    
collect_user_details()