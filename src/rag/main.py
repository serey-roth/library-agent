import uuid

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
from logger import logger
from rag.ingestion import load_locations, load_authors, load_books, check_books_exist


logger.debug("Starting application")
logger.debug("Loading secrets")
try:
    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
    logger.debug("Database configuration loaded")
except KeyError:
    logger.error("Database configuration missing - one or more secrets failed to load")
    
    
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    st.error("❌ Database configuration missing. Please check your environment variables.")
    st.stop()


def show_status(message: str, type: str = "info"):
    if type == "success":
        st.markdown(f'<div class="status-container"><div class="success-text">✅ {message}</div></div>', unsafe_allow_html=True)
    elif type == "error":
        st.markdown(f'<div class="status-container"><div class="error-text">❌ {message}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-container"><div class="loading-text">ℹ️ {message}</div></div>', unsafe_allow_html=True)


def ingest_data_with_progress(driver: GraphDatabase.driver):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        logger.debug("Loading library branches...")
        status_text.text("🔄 Loading library branches...")
        progress_bar.progress(0.1)
        load_locations(driver)
        
        logger.debug("Loading authors...")
        status_text.text("🔄 Loading authors...")
        progress_bar.progress(0.35)
        load_authors(driver)
        
        logger.debug("Loading books...")
        status_text.text("🔄 Loading books...")
        progress_bar.progress(0.6)
        load_books(driver)
        
        logger.debug("Finalizing collection...")
        status_text.text("🔄 Finalizing collection...")
        progress_bar.progress(0.85)
        
        status_text.text("✅ Data ingestion completed!")
        progress_bar.progress(1.0)
        
        progress_bar.empty()
        status_text.empty()
        
        logger.debug("Data ingestion completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        if 'progress_bar' in locals() and 'status_text' in locals():
            status_text.text(f"❌ Setup failed: {str(e)}")
            progress_bar.progress(1.0)
        show_status(f"Setup failed: {str(e)}", "error")
        raise
    

@st.cache_resource
def initialize_system() -> tuple[CompiledStateGraph, RunnableConfig, GraphDatabase.driver]:
    try:
        with st.spinner("🔗 Connecting to database..."):
            driver = GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH)
        
        needs_ingestion = not check_books_exist(driver)
        if needs_ingestion:
            logger.debug("Starting data ingestion...")
            ingest_data_with_progress(driver)
        else:
            logger.debug("Data already exists - skipping data ingestion")
        
        with st.spinner("🧠 Initializing AI assistant..."):
            logger.debug("Creating graph...")
            graph = LangchainNeo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                refresh_schema=False,
            )
        
            logger.debug("Creating embeddings...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            logger.debug("Creating vectorstore...")
            vectorstore = Neo4jVector.from_existing_graph(
                embedding=embeddings,
                node_label="Book",
                embedding_node_property="embedding",
                text_node_properties=["norm_desc"],
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
            )
        
            logger.debug("Creating agent...")
            llm = ChatOpenAI(model="gpt-4o-mini")
            tools = [create_search_books_tool(graph, vectorstore, embeddings)]
            agent = create_react_agent(
                model=llm, 
                tools=tools, 
                prompt=SYSTEM_PROMPT, 
                checkpointer=InMemorySaver()
            )
            thread_id = str(uuid.uuid4())
            config = RunnableConfig(configurable={"thread_id": thread_id})
        
        logger.debug("AI system initialized")
        
        return agent, config, driver
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        show_status(f"Initialization failed: {str(e)}", "error")
        st.stop()


def chat_with_agent(message: str, agent, config):   
    logger.debug(f"User query: {message}")
    
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
    
    try:
        for response in _gather_agent_responses():
            yield response
    except Exception as e:
        logger.error(f"Chat with agent failed: {str(e)}")
        yield f"Sorry, I encountered an error: {str(e)}"
        
        
SAMPLE_USER_PROMPTS = [
    "Find some gothic fiction novels",
    "What can I find at Queen Anne?",
    "Recommend a good romance novel",
    "Find some romance novels at Queen Anne"
]


def main():
    # Experimental demo banner
    st.markdown("""
    <div style="background-color: #ff6b6b; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center;">
        <strong>⚠️ EXPERIMENTAL DEMO</strong><br>
        This demo project is not affiliated with Seattle Public Library.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Seattle Public Library AI Assistant</h1>', unsafe_allow_html=True)
    
    agent, config, driver = initialize_system()
    
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