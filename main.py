import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq 
from langchain_classic.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_classic.tools import Tool,tool
from langchain_classic import hub
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


st.set_page_config(page_title="JFK Assistant", layout="wide")
# --- 1. Vector DB Setup ---
# Using @st.cache_resource so the DB doesn't reload on every single click
@st.cache_resource
def load_pdf_into_vector_db():
    load_dotenv()
    embedding_model = CohereEmbeddings(model='embed-multilingual-light-v3.0')
    vectordb = FAISS.load_local(
        folder_path='vector_db',
        index_name='john_f_kennedy_vector_db',
        allow_dangerous_deserialization=True,
        embeddings=embedding_model
    )
    return vectordb.as_retriever(search_kwargs={'k': 3})
    
retriever = load_pdf_into_vector_db()

@tool
def agent_rag_tool(query: str): 
    """Useful for answering questions about John F. Kennedy's life, presidency, and history."""
    docs = retriever.invoke(query) 
    return "\n".join([doc.page_content for doc in docs])

# --- 2. Agent Configuration ---
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

template = PromptTemplate.from_template(template=prompt_template)
llm2 = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
llm=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash-lite')

def create_agent_executor():
    tools = [agent_rag_tool]
    our_agent = create_react_agent(llm=llm, prompt=template, tools=tools)
    return AgentExecutor(
        agent=our_agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

agent_executor = create_agent_executor()

# --- 3. Streamlit UI & Session State ---

st.title("JFK Research Assistant")

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("When did JFK become president?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        response = agent_executor.invoke(
            {"input": query},
            {"callbacks": [st_callback]}
        )
        
        final_answer = response["output"]
        st.write(final_answer)
        
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})