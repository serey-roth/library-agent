import os
import uuid
import time
from dotenv import load_dotenv

from neo4j import GraphDatabase

from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores.neo4j_vector import Neo4jVector    
from langchain_neo4j import Neo4jGraph as LangchainNeo4jGraph
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

import streamlit as st

from rag.tools import create_search_books_tool
from rag.prompts import SYSTEM_PROMPT

from rag.ingestion import check_books_exist, load_locations, load_authors, load_books

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    st.error("❌ Database configuration missing. Please check your environment variables.")
    st.stop()

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

thread_id = str(uuid.uuid4())

def show_loading_state(message: str, progress: float = None):
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.spinner()
        with col2:
            st.markdown(f'<div class="loading-text">{message}</div>', unsafe_allow_html=True)
        if progress is not None:
            st.progress(progress)

def show_status(message: str, type: str = "info"):
    if type == "success":
        st.markdown(f'<div class="status-container"><div class="success-text">✅ {message}</div></div>', unsafe_allow_html=True)
    elif type == "error":
        st.markdown(f'<div class="status-container"><div class="error-text">❌ {message}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-container"><div class="loading-text">ℹ️ {message}</div></div>', unsafe_allow_html=True)

def ingest_data_with_progress(driver: GraphDatabase.driver):
    try:
        show_loading_state("Loading library branches...", 0.1)
        load_locations(driver)
        time.sleep(0.2)
        
        show_loading_state("Loading authors...", 0.3)
        load_authors(driver)
        time.sleep(0.2)
        
        show_loading_state("Loading books...", 0.6)
        load_books(driver)
        time.sleep(0.2)
        
        show_loading_state("Finalizing collection...", 0.9)
        time.sleep(0.3)
        
        return True
        
    except Exception as e:
        show_status(f"Setup failed: {str(e)}", "error")
        raise

@st.cache_resource
def initialize_system() -> tuple[CompiledStateGraph, RunnableConfig, GraphDatabase.driver]:
    try:
        with st.spinner("Connecting to database..."):
            driver = GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH)
            
        needs_ingestion = not check_books_exist(driver)
        if needs_ingestion:
            with st.spinner("Setting up library collection..."):
                ingest_data_with_progress(driver)
        
        with st.spinner("Initializing AI assistant..."):
            graph = LangchainNeo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                refresh_schema=False,
            )
        
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = Neo4jVector.from_existing_graph(
                embedding=embeddings,
                node_label="Book",
                embedding_node_property="embedding",
                text_node_properties=["norm_desc"],
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
            )
        
            llm = ChatOpenAI(model="gpt-4o-mini")
            tools = [create_search_books_tool(graph, vectorstore, embeddings)]
            agent = create_react_agent(
                model=llm, 
                tools=tools, 
                prompt=SYSTEM_PROMPT, 
                checkpointer=InMemorySaver()
            )
            config = RunnableConfig(configurable={"thread_id": thread_id})
        
        return agent, config, driver
        
    except Exception as e:
        show_status(f"Initialization failed: {str(e)}", "error")
        st.stop()

def chat_with_agent(message: str, agent, config):   
    def _gather_agent_responses():
        responses = []
        for msg, metadata in agent.stream({"messages": [{
            "role": "user",
            "content": message
        }]}, stream_mode="messages", config=config):
            langgraph_node = metadata.get("langgraph_node", "")
            if langgraph_node == "agent" and hasattr(msg, 'content') and msg.content:
                responses.append(msg.content)
        return responses
    
    for response in _gather_agent_responses():
        yield response
        
SAMPLE_USER_PROMPTS = [
    "Find gothic fiction books",
    "What can I find at Queen Anne?",
    "Books by Stephen King?",
    "Recommend a good romance novel"
]

def main():
    st.markdown('<h1 class="main-header">Seattle Public Library AI Assistant</h1>', unsafe_allow_html=True)
    
    try:
        agent, config, driver = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "has_chat_started" not in st.session_state:
        st.session_state.has_chat_started = False
    
    if not st.session_state.has_chat_started:
       with st.container():
            with st.chat_message("assistant"):
                st.markdown("Hi there! I can help you find books at different Seattle library branches, and discover your next great read. So how can I help you today?")

            col1, col2 = st.columns(2)
            with col1:
                for i, prompt in enumerate(SAMPLE_USER_PROMPTS[:len(SAMPLE_USER_PROMPTS)//2]):
                    if st.button(prompt, use_container_width=True, key=f"prompt_left_{i}"):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.has_chat_started = True
                        st.rerun()
            with col2:
                for i, prompt in enumerate(SAMPLE_USER_PROMPTS[len(SAMPLE_USER_PROMPTS)//2:], start=len(SAMPLE_USER_PROMPTS)//2):
                    if st.button(prompt, use_container_width=True, key=f"prompt_right_{i}"):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.has_chat_started = True
                        st.rerun()
    
    if st.session_state.has_chat_started:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
    if user_message := st.chat_input("What's on your mind?"):
        st.session_state.has_chat_started = True
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.rerun()
        
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_message = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.write_stream(chat_with_agent(user_message, agent, config))
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()