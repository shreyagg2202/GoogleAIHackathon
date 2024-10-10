# semantic_search.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the model and keyphrase embeddings when the module is imported
Semantic_Search_model = SentenceTransformer('all-mpnet-base-v2')

threshold = 0.65
Keyphrases = [
    "I want to select this policy", "I want to go ahead with this policy", 
    "I want to choose this policy", "Let's proceed with this policy", "Sign me up for this policy", "I'd like to buy this policy",
    "This policy suits my needs", "This is my preferred policy",
    "I want to purchase this policy", "I'm ready to enroll in this policy",
    "This policy works for me", "I have decided on this policy", "Please proceed with this policy",
    "I want to move forward with this policy", "This is the policy I'd like", "Enroll me in this policy",
    "I am interested in purchasing this policy",
    "I want to select this plan", "I want to go ahead with this plan", 
    "I want to choose this plan", "Let's proceed with this plan", "Sign me up for this plan", "I'd like to buy this plan",
    "This plan suits my needs", "This is my preferred plan",
    "I want to purchase this plan", "I'm ready to enroll in this plan",
    "This plan works for me", "I have decided on this plan", "Please proceed with this plan",
    "I want to move forward with this plan", "This is the plan I'd like", "Enroll me in this plan",
    "I am interested in purchasing this plan"
]

Keyphrase_embeddings = Semantic_Search_model.encode(Keyphrases)

def is_similar(user_input, threshold=0.75):
    """
    Determines if the user's input is similar to any of the keyphrases.

    Parameters:
    - user_input (str): The user's input text.
    - threshold (float): The similarity threshold.

    Returns:
    - bool: True if similar, False otherwise.
    """
    user_input_embedding = Semantic_Search_model.encode([user_input])
    similarities = cosine_similarity(user_input_embedding, Keyphrase_embeddings)[0]
    max_similarity = np.max(similarities)
    return max_similarity >= threshold
