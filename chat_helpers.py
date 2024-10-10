# chat_helpers.py

import streamlit as st

def assistant_response_format(content, MODEL_ROLE='ai', AI_AVATAR_ICON='✨'):
    with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
        st.markdown(content)
    st.session_state.messages.append({
        'role': MODEL_ROLE,
        'content': content,
        'avatar': AI_AVATAR_ICON,
    })

def user_response_format(prompt):
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt,
    })
