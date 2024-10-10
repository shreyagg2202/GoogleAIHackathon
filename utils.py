# utils.py

import google.generativeai as genai
from chromadb.api.types import Documents, Embeddings
from chromadb.api import EmbeddingFunction

# Define the embedding function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004'
        return genai.embed_content(model=model, content=input, task_type="question_answering")["embedding"]

def get_relevant_passage(query_embedding, db):
    results = db.query(query_embeddings=[query_embedding], n_results=3)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    return documents, metadatas

def make_prompt(query, db, embedding_function):
    query_embeddings = embedding_function(query)
    documents, metadatas = get_relevant_passage(query_embeddings[0], db)
    relevant_passage = []

    # Combine metadata with document content
    for doc, meta in zip(documents, metadatas):
        policy_name = meta.get('policy_name', 'Unknown Policy')
        combined = f"Policy Name: {policy_name}\n{doc}"
        relevant_passage.append(combined)
    
    relevant_passage = "\n\n---\n\n".join(relevant_passage)
    prompt = f"""
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
    """
    return prompt

def detect_policy_type(conversation):

    # Convert the conversation (list of messages) into a single string
    conversation_text = ""
    for message in conversation:
        role = message['role']
        content = message['content']
        if role == 'user':
            conversation_text += f"User: {content}\n"
        elif role == 'ai':
            conversation_text += f"Assistant: {content}\n"


    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""You are a bot which analyses a conversation and returns the specific name and type of the policy that the user has chosen. 
        You must return only the policy name which you think is most likely selected. If the user has not selected a policy, return "None".
        Do not return any other text, or other details. Dont even say Hi or anything.

        Your response should be in the format:
        Policy Name = "policy_name"
        Policy Type = "policy_type" (ex - health insurance, vehicle insurance, life insurance)
        """,
    generation_config=generation_config,
    )

    selected_policy = model.generate_content(conversation_text)
    # Parse the LLM's response to extract policy name and type
    try:
        policy_name = selected_policy.text.split('Policy Name = ')[1].split('"')[1]
        policy_type = selected_policy.text.split('Policy Type = ')[1].split('"')[1]
    except IndexError:
        policy_name = "None"
        policy_type = "None"
        
    return policy_name, policy_type

# Function to save details to GitHub
import json
import base64
import requests
import datetime
import streamlit as st

def save_details_to_github(user_details):
    # Get GitHub credentials from st.secrets
    GITHUB_TOKEN = st.secrets['github_token']
    GITHUB_REPO = st.secrets['github_repo']  # In the format 'username/repo_name'

    content = json.dumps(user_details.dict(), indent=4)
    # Create a unique filename using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"user_details_{timestamp}.json"

    # Prepare the API request
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/user_data/{filename}"
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    message = f"Add user details {filename}"
    content_base64 = base64.b64encode(content.encode()).decode()
    payload = {
        "message": message,
        "content": content_base64
    }

    response = requests.put(url, headers=headers, data=json.dumps(payload))
