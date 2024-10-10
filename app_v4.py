#Imports
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import google.generativeai as genai
import streamlit as st
import time
import chromadb
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_google_genai import ChatGoogleGenerativeAI

# Import modules
from personal_details import (
    BasePersonalDetails,
    VehiclePersonalDetails,
    HealthPersonalDetails,
    LifePersonalDetails,
    check_what_is_empty,
    add_non_empty_details,
)
from utils import (
    GeminiEmbeddingFunction,
    make_prompt,
    detect_policy_type,
    save_details_to_github,
)
from chat_helpers import assistant_response_format, user_response_format
from semantic_search import is_similar

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
if 'PersonalDetails' not in st.session_state:
    st.session_state.PersonalDetails = BasePersonalDetails
if 'selected_policy_name' not in st.session_state:
    st.session_state.selected_policy_name = None
if 'selected_policy_type' not in st.session_state:
    st.session_state.selected_policy_type = None

#initialization
API_KEY = st.secrets["gemini_api_key"]
updated_details = {}

MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

st.set_page_config(
    page_title="Policy Bazaar",
    page_icon="🔥"
)

try:
    genai.configure(api_key=API_KEY)
except AttributeError as e:
    st.warning("API Key not working")

embedding_function = GeminiEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="Chroma_DB/")
db = chroma_client.get_collection(name="Test3", embedding_function=embedding_function)

st.title("PolicyPal")
st.caption("A policy advisor powered by Google Gemini")

# Instructions for the user
st.markdown("""
### Welcome to Policy Bazaar!

This assistant is here to help you understand and select the best insurance policy for your needs. You can ask any questions about insurance policies, and the assistant will provide you with expert advice.

**How to use this assistant:**
- Start by asking any questions you have about insurance policies.
- If you decide to select a policy, the assistant will guide you through the process.
- You will be asked to confirm your selection.
- After confirmation, you will be prompted to provide some personal details.
- You can review and edit your details before submitting.

Feel free to ask any questions!

---
""")

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
    user_response_format(prompt)

    if st.session_state.conversation_phase == 'policy_selection':
        if is_similar(prompt):
            # Detected that the user wants to select a policy
            st.session_state.conversation_phase = 'awaiting_confirmation'
            # Assistant asks for confirmation
            
            confirmation_message = "Are you sure you want to select this policy?"
            assistant_response_format(confirmation_message)
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
            current_conversation = st.session_state.messages
            policy_name, policy_type = detect_policy_type(current_conversation)
            st.session_state.selected_policy_name = policy_name
            st.session_state.selected_policy_type = policy_type


            # Update `ask_for` list and `PersonalDetails` class
            if 'vehicle' in policy_type:
                st.session_state.PersonalDetails = VehiclePersonalDetails
                st.session_state.ask_for = [
                    'name', 'age', 'date_of_birth', 'address', 'phone_number', 'email_address',
                    'vehicle_age', 'vehicle_type', 'previous_accidents'
                ]
            elif 'health' in policy_type:
                st.session_state.PersonalDetails = HealthPersonalDetails
                st.session_state.ask_for = [
                    'name', 'age', 'date_of_birth', 'address', 'phone_number', 'email_address',
                    'allergies', 'current_medications', 'occupation','income'
                ]
            elif 'life' in policy_type:
                st.session_state.PersonalDetails = LifePersonalDetails
                st.session_state.ask_for = [
                    'name', 'age', 'date_of_birth', 'address', 'phone_number', 'email_address',
                    'height', 'weight','chronic_illnesses', 'occupation', 'income'
                ]
            else:
                st.session_state.PersonalDetails = BasePersonalDetails
                st.session_state.ask_for = [
                    'name', 'age', 'date_of_birth', 'address', 'phone_number', 'email_address'
                ]

            st.session_state.user_details = st.session_state.PersonalDetails()

            # Assistant acknowledges and proceeds
            confirmation_ack = (
                "Thanks for selecting a policy and trusting us. "
                "I will now ask you a set of questions to gather your details "
                "for further processing of the selected policy."
            )
            assistant_response_format(confirmation_ack)

            # Ask the first question
            question = "Please provide your name"
            assistant_response_format(question)
            
        elif user_response in ['no', 'n']:
            st.session_state.policy_selected = False
            st.session_state.conversation_phase = 'policy_selection'
            # Assistant continues conversation
            continue_message = "Okay, let's continue our conversation."
            assistant_response_format(continue_message)
        else:
            # Assistant prompts for a valid response
            invalid_response = "Please respond with 'yes' or 'no'."
            assistant_response_format(invalid_response)

    elif st.session_state.conversation_phase == 'collecting_details':
        # Collecting user details
        if st.session_state.ask_for:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_KEY)

            def filter_response(text_input, user_details):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=API_KEY)
                chain = llm.with_structured_output(st.session_state.PersonalDetails)
                res = chain.invoke(text_input)
                user_details = add_non_empty_details(user_details, res)
                ask_for = check_what_is_empty(user_details)
                return user_details, ask_for

            st.session_state.user_details, st.session_state.ask_for = filter_response(prompt, st.session_state.user_details)

            if st.session_state.ask_for:
                # Prepare the list of remaining items
                remaining_items = ', '.join(st.session_state.ask_for)

                # Define the system message with the remaining items
                system_message_content = f"""
        You are an assistant that needs to collect the following information from the user: {remaining_items}.
        - Ask for one item at a time in a conversational manner.
        - Dont greet the user or act like this is the first question. Straight up ask the question.
        """
                # Create the prompt template
                prompt_temp = ChatPromptTemplate(
                    [
                        SystemMessage(content=system_message_content),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                
                # Build the chain by combining the prompt and the LLM
                chain = prompt_temp | llm

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
                assistant_response_format(next_question)
            else:
                # All details collected
                st.session_state.conversation_phase = 'finished'

        else:
            # All details collected
            st.session_state.conversation_phase = 'finished'

if st.session_state.conversation_phase == 'finished':
    # Conversation is finished, you can reset or handle further interactions
    st.write("Please review your details below. You can make changes if necessary before submitting.")

    # Create a form for editing the details
    with st.form(key='details_form'):
        for field_name, field_value in st.session_state.user_details.dict().items():
            input_label = field_name.replace('_',' ').title()
            updated_value = st.text_input(input_label, value=field_value)
            updated_details[field_name] = updated_value

        # Submit button
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        for field_name, updated_value in updated_details.items():
            setattr(st.session_state.user_details, field_name, updated_value)

        # Save details to GitHub
        save_details_to_github(st.session_state.user_details)
        
        # Thank the user
        st.success("Thank you! Your details have been submitted.")
        # Reset the conversation
        st.session_state.conversation_phase = 'policy_selection'
        st.session_state.policy_selected = False
        st.session_state.user_details = None
        st.session_state.ask_for = []
        st.session_state.messages = []

        # Stop the app
        st.stop()
            
    