from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
import re, os

import streamlit as st


# Load the env file
load_dotenv(override=True)

# Initialize the LLM
llm_qwen = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:latest",
    temperature=0.5,
    max_tokens=400
)

db_name = "chathistory.db"

# Make sure the Database is newly created for every run
if "db_initialized" not in st.session_state:
    if os.path.exists(db_name):
        os.remove(db_name)
    st.session_state["db_initialized"] = True

# Making remove_think_block as a Runnable in order to chain it
@chain
def remove_think_block(text) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///" + db_name)


###################### Streamlit code Start ######################

st.title("How can I help you ?")
prompt = st.chat_input("Enter your query here")

session_id_old = session_id = "General"

def update_session():
    st.session_state.chat_history = []
    get_session_history(session_id_old).clear()

session_id = st.text_input("Enter your session topic here:", session_id, key="session", on_change=update_session)

if st.button("Start a new session"):
    st.session_state.chat_history = []
    get_session_history(session_id_old).clear()

# Updating session_id_old for next iterations
session_id_old = session_id 

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

###################### Streamlit code End ######################


prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are like Wikipedia. Provide concise and correct responses."),
    ('human', "{prompt} /no_think"),
    ('placeholder', "{history}")
])

chain = prompt_template | llm_qwen

def invoke_history(chain, session_id, prompt):

    history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="prompt",
        output_messages_key=None
    )

    for response in history.stream({"prompt": prompt},
        config={"configurable": {"session_id": session_id}}):
        yield response


###################### Streamlit code Start ######################

if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message('assistant'):
        output = st.write_stream(invoke_history(chain, session_id, prompt))
    
    st.session_state.chat_history.append({'role': 'assistant', 'content': output})

###################### Streamlit code End ######################
