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


# Initialize session state variables if not already set
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []
if 'conversation_phase' not in st.session_state:
    st.session_state.conversation_phase = 'policy_selection'  # Initial phase
if 'policy_selected' not in st.session_state:
    st.session_state.policy_selected = False
if 'user_details' not in st.session_state:
    st.session_state.user_details = None
if 'ask_for' not in st.session_state:
    st.session_state.ask_for = ['name', 'date_of_birth', 'address', 'phone_number', 'email_address']

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
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_KEY)
    chain = llm.with_structured_output(PersonalDetails)
    res = chain.invoke(text_input)
    user_details = add_non_empty_details(user_details,res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for

st.session_state.model = model
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(
#         name=message['role'],
#         avatar=message.get('avatar'),
#     ):
#         st.markdown(message['content'])


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
    if st.session_state.conversation_phase == 'policy_selection':
        if is_similar(prompt):
            # Detected that the user wants to select a policy
            st.session_state.conversation_phase = 'awaiting_confirmation'
            # Assistant asks for confirmation
            confirmation_message = "Are you sure you want to select this policy?"
            with st.chat_message(
                 name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
                ):
                    st.markdown(confirmation_message)
            st.session_state.messages.append(
                dict(
                    role=MODEL_ROLE,
                    avatar=AI_AVATAR_ICON,
                )
            )
        else:
            # Continue policy selection conversation with the LLM
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
                    content=full_response,
                    avatar=AI_AVATAR_ICON,
                )
            )
            st.session_state.gemini_history = st.session_state.chat.history

    elif st.session_state.conversation_phase == 'awaiting_confirmation':
        # Waiting for user's confirmation response
        user_response = prompt.lower().strip()
        if user_response in ['yes', 'y']:
            st.session_state.policy_selected = True
            st.session_state.conversation_phase = 'collecting_details'
            # Assistant acknowledges and proceeds
            confirmation_ack = (
                "Thanks for selecting a policy and trusting us. "
                "I will now ask you a set of questions to gather your details "
                "for further processing of the selected policy."
            )
            with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
                ):
                    st.markdown(confirmation_ack)
            st.session_state.messages.append(
                dict(
                    role=MODEL_ROLE,
                    content=confirmation_ack,
                    avatar=AI_AVATAR_ICON,
                )
            )
            st.session_state.user_details = PersonalDetails()
            question = "Please provide your name"
            with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
                ):
                    st.markdown(question)
            st.session_state.messages.append(
                dict(
                    role=MODEL_ROLE,
                    content=question ,
                    avatar=AI_AVATAR_ICON,
                )
            )
        elif user_response in ['no', 'n']:
            st.session_state.policy_selected = False
            st.session_state.conversation_phase = 'policy_selection'
            # Assistant continues conversation
            continue_message = "Okay, let's continue our conversation."
            with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
                ):
                    st.markdown(continue_message)
                    st.session_state.messages.append(
                        dict(
                            role=MODEL_ROLE,
                            content=continue_message,
                            avatar=AI_AVATAR_ICON,
                        )
                    )
        else:
            # Assistant prompts for a valid response
            invalid_response = "Please respond with 'yes' or 'no'."
            with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
                ):
                    st.markdown(invalid_response)
            st.session_state.messages.append(
                dict(
                    role=MODEL_ROLE,
                    content=invalid_response,
                    avatar=AI_AVATAR_ICON,
                )
            )

    elif st.session_state.conversation_phase == 'collecting_details':
        # Collecting user details
        # Use your LLM to parse the user's response and update details
        ask_for = st.session_state.ask_for
        while ask_for:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_KEY)

            user_details, ask_for = filter_response(prompt, st.session_state.user_details)       
            # user_details = collect_user_details()

            st.session_state.user_details = user_details
            st.session_state.ask_for = ask_for

            if st.session_state.ask_for:
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
                
                next_question = ai_message.content  # Store the next question for the user to response
                with st.chat_message(
                    name=MODEL_ROLE,
                    avatar=AI_AVATAR_ICON,
                    ):
                        st.markdown(next_question)
                        st.session_state.messages.append(
                            dict(
                                role=MODEL_ROLE,
                                content=next_question,
                                avatar=AI_AVATAR_ICON,
                            )
                        )
            else:
                # All details collected
                st.session_state.conversation_phase = 'finished'
                thank_you_message = (
                "Thank you! Your details have been submitted. "
                "Our customer support team will contact you soon."
            )
                with st.chat_message(
                    name=MODEL_ROLE,
                    avatar=AI_AVATAR_ICON,
                    ):
                        st.markdown(thank_you_message)
                        st.session_state.messages.append(
                            dict(
                                role=MODEL_ROLE,
                                content=thank_you_message,
                                avatar=AI_AVATAR_ICON,
                            )
                        )

        # Check if the user wants to restart
        # Not completed yet
        if prompt.strip().lower() == 'restart':
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
            start_over_message = "Assistant: Let's start over. Please provide your details again."
            with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
                ):
                    st.markdown(start_over_message)
                    st.session_state.messages.append(
                        dict(
                            role=MODEL_ROLE,
                            content=start_over_message,
                            avatar=AI_AVATAR_ICON,
                        )
                    )
                # Optionally, clear the conversation history
            messages = []

    elif st.session_state.conversation_phase == 'finished':
        # Conversation is finished, you can reset or handle further interactions
        end_message = "If you have any more questions, feel free to ask!"
        with st.chat_message(
                    name=MODEL_ROLE,
                    avatar=AI_AVATAR_ICON,
                    ):
                        st.markdown(end_message)
                        st.session_state.messages.append(
                            dict(
                                role=MODEL_ROLE,
                                content=end_message,
                                avatar=AI_AVATAR_ICON,
                            )
                        )
        st.session_state.conversation_phase = 'policy_selection'  # Reset to initial phase
        st.session_state.policy_selected = False
        st.session_state.user_details = None
        st.session_state.ask_for = ['name', 'date_of_birth', 'address', 'phone_number', 'email_address']
