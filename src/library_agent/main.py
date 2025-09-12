from dotenv import load_dotenv  

load_dotenv()

from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import Tool
import pandas as pd
from haystack.components.generators.chat import OpenAIChatGenerator
import gradio as gr
from haystack.components.tools import ToolInvoker
from haystack.components.generators.utils import print_streaming_chunk

def process_excel_rows_to_documents(file_path: str) -> list[Document]:
    """Process Excel file and create one document per row with title and summary."""
    df = pd.read_excel(file_path, sheet_name="books")
    documents = []
    
    for index, row in df.iterrows():
        title = str(row.get('title', '')) if 'title' in row else ''
        subtitle = str(row.get('subtitle', '')) if 'subtitle' in row else ''
        norm_desc = str(row.get('normalized_description', '')) if 'normalized_description' in row else ''
        categories = str(row.get('categories', '')) if 'categories' in row else ''
        authors = str(row.get('authors', '')) if 'authors' in row else ''
        publisher = str(row.get('publisher', '')) if 'publisher' in row else ''
        published_date = str(row.get('published_date', '')) if 'published_date' in row else ''
        isbn = str(row.get('isbn', '')) if 'isbn' in row else ''   
        locations = str(row.get('locations', '')) if 'locations' in row else ''
        
        if (not norm_desc) or (not title) or (not authors):
            continue
        
        content = f"\"{title}, {subtitle}\" by {authors}\n{norm_desc}"
                
        doc = Document(
            content=content,
            meta={
                'row_index': index,
                'title': title,
                'subtitle': subtitle,
                'authors': authors,
                'normalized_description': norm_desc,
                'categories': categories,
                'publisher': publisher,
                'published_date': published_date,
                'isbn': isbn,
                'locations': locations,
            }
        )
        documents.append(doc)

    return documents


def create_ingestion_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    """Create an ingestion pipeline for the document store."""
    document_cleaner = DocumentCleaner()
    document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
    document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    
    ingestion_pipeline = Pipeline()
    ingestion_pipeline.add_component("cleaner", document_cleaner)
    ingestion_pipeline.add_component("embedder", document_embedder)
    ingestion_pipeline.add_component("writer", document_writer)
    ingestion_pipeline.connect("cleaner", "embedder")
    ingestion_pipeline.connect("embedder", "writer")
    return ingestion_pipeline


def create_embedding_prompt_builder() -> ChatPromptBuilder:
    """Create a prompt builder for the embeddings"""
    embedding_prompt_template = [
        ChatMessage.from_system(
            """You are a helpful library assistant. You can ONLY answer with books that are explicitly mentioned in the provided context below. 
            
            IMPORTANT RULES:
            - ONLY answer with books that appear in the context provided. If no relevant books are found in the context, say "no_books_found"
            - Always include the book title, author, ISBN, locations, and description for each book
            - Be specific about why each book matches the user's request
            - ALWAYS explain why the book matches the user's request"""
        ),
        ChatMessage.from_user(
            """Based on the following book collection, answer the user's query. Only answer with books that are explicitly listed here.

    Available Books:
    {% for document in documents %}
        Title: {{ document.meta.title }}
        Author(s): {{ document.meta.authors }}
        Description: {{ document.content }}
        Categories: {{ document.meta.categories }}
        ISBN: {{ document.meta.isbn }}
        Locations: {{ document.meta.locations }}
        Publisher: {{ document.meta.publisher }}
        Published Date: {{ document.meta.published_date }}
        ---
    {% endfor %}

    User Query: {{query}}"""
        )
    ]
    
    return ChatPromptBuilder(template=embedding_prompt_template, required_variables=["query", "documents"])
    
    
def create_websearch_prompt_builder() -> ChatPromptBuilder:
    """Create a prompt builder for the websearch results."""
    prompt_template_web_search = [
        ChatMessage.from_system(
            """You are a helpful library assistant. You can ONLY answer with books that are explicitly mentioned in the provided context below. 
            
            IMPORTANT RULES:
            - ONLY answer with books that appear in the context provided
            - NEVER invent, create, or answer with books that are not in the context
            - If no relevant books are found in the context, say "no_books_found"
            - Always include the book title, author, ISBN, and a summary of the book
            - The summary must be FACTUAL and OBJECTIVE - only state what happens in the book, not opinions or judgments
            - Use neutral, descriptive language - avoid words like "compelling", "engaging", "thrilling", "excellent", etc.
            - Focus on plot facts, character names, settings, and events - not subjective assessments
            - If no relevant books are found in the context, say "no_books_found"
            - ALWAYS explain why the book matches the user's request
        """),
        ChatMessage.from_user(
            """Based on these books found on the web, answer the user's query.

    Search Results:
    {% for document in documents %}
        Title: {{ document.meta.title }}
        Author(s): {{ document.meta.authors }}
        Description: {{ document.content }}
        ISBN: {{ document.meta.isbn }}
        Publisher: {{ document.meta.publisher }}
        Published Date: {{ document.meta.published_date }}
        ---
    {% endfor %}

    User Query: {{query}}"""
        )
    ]
    
    return ChatPromptBuilder(template=prompt_template_web_search, required_variables=["query", "documents"])


def create_routes() -> list[dict]:
    """Create routes for the agentic RAG pipeline."""
    routes = [
        {
            "condition": "{{'no_books_found' in replies[0].text}}",
            "output": "{{query}}",
            "output_name": "go_to_websearch",
            "output_type": str,
        },
        {
            "condition": "{{'no_books_found' not in replies[0].text}}",
            "output": "{{replies[0].text}}",
            "output_name": "answer",
            "output_type": str,
        },
    ]
    return routes


def create_agentic_rag_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    """Create an agentic RAG pipeline."""
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    embedding_retriever = InMemoryEmbeddingRetriever(
        document_store=document_store, 
        top_k=10, 
        scale_score=False)
    embedding_prompt_builder = create_embedding_prompt_builder()
    websearch_prompt_builder = create_websearch_prompt_builder()
    embedding_llm = OpenAIChatGenerator(
        model="gpt-4o-mini",
    )
    websearch_llm = OpenAIChatGenerator(
        model="gpt-4o-mini",
    )
    websearch = SerperDevWebSearch(
        top_k=5,  # Limit to top 5 results
        search_params={
            "num": 5,  # Number of results
            "gl": "us",  # Country
            "hl": "en",  # Language
        }
    )
    router = ConditionalRouter(routes=create_routes())
    
    agentic_rag_pipeline = Pipeline()
    agentic_rag_pipeline.add_component("text_embedder", text_embedder)
    agentic_rag_pipeline.add_component("embedding_retriever", embedding_retriever)
    agentic_rag_pipeline.add_component("embedding_prompt_builder", embedding_prompt_builder)
    agentic_rag_pipeline.add_component("embedding_llm", embedding_llm)
    agentic_rag_pipeline.add_component("websearch", websearch)
    agentic_rag_pipeline.add_component("websearch_prompt_builder", websearch_prompt_builder)
    agentic_rag_pipeline.add_component("websearch_llm", websearch_llm)
    agentic_rag_pipeline.add_component("router", router)
    
    # Embedding RAG
    agentic_rag_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    agentic_rag_pipeline.connect("embedding_retriever", "embedding_prompt_builder.documents")
    agentic_rag_pipeline.connect("embedding_prompt_builder.prompt", "embedding_llm.messages")
    agentic_rag_pipeline.connect("embedding_llm.replies", "router.replies")
    
    # Fallback to websearch
    agentic_rag_pipeline.connect("router.go_to_websearch", "websearch.query")
    agentic_rag_pipeline.connect("router.go_to_websearch", "websearch_prompt_builder.query")
    agentic_rag_pipeline.connect("websearch.documents", "websearch_prompt_builder.documents")
    agentic_rag_pipeline.connect("websearch_prompt_builder", "websearch_llm")
    
    return agentic_rag_pipeline


def create_find_books_tool(document_store: InMemoryDocumentStore) -> Tool:
    """Create an agentic RAG pipeline as a tool."""
    def agentic_rag_pipeline_as_tool(query: str) -> str:
        """Run the agentic RAG pipeline as a tool."""
        agentic_rag_pipeline = create_agentic_rag_pipeline(document_store)
        result = agentic_rag_pipeline.run({"text_embedder": {"text": query}, "embedding_prompt_builder": {"query": query}, "router": {"query": query}})
        if ("router" in result):    
            return result["router"]["answer"]
        elif ("embedding_llm" in result):
            return result["embedding_llm"]["replies"][0].text
        else:
            return result["websearch_llm"]["replies"][0].text
    
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to use in the search. Infer this from the user's message. It should be a question or a statement",
            }
        },
        "required": ["query"],
    }

    return Tool(
        name="find_books",
        description="Search for books in the library collection using semantic search. If no relevant books are found locally, it will search the web for additional book recommendations.",
        parameters=parameters,
        function=agentic_rag_pipeline_as_tool,
    )

response = None
messages = [
    ChatMessage.from_system(
        "Use the tool that you're provided with. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    )
]


def create_chat_agent_demo(document_store: InMemoryDocumentStore) -> gr.ChatInterface:
    """Create a chat agent demo."""
    tools = [create_find_books_tool(document_store)]
    tool_invoker = ToolInvoker(tools=tools)
    chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", tools=tools, streaming_callback=print_streaming_chunk)
    
    def chatbot_with_tc(message, history):
        global messages
        messages.append(ChatMessage.from_user(message))
        response = chat_generator.run(messages=messages)

        while True:
            # if OpenAI response is a function call
            if response and response["replies"][0].tool_calls:
                tool_result_messages = tool_invoker.run(messages=response["replies"])["tool_messages"]
                messages = messages + response["replies"] + tool_result_messages
                response = chat_generator.run(messages=messages)

            # Regular Conversation
            else:
                messages.append(response["replies"][0])
                break
        return response["replies"][0].text

    return gr.ChatInterface(
        fn=chatbot_with_tc,
        type="messages",
        examples=[
            "Help me find a book by Rachel Hawkins",
            "Help me find a book about the history of the city of Seattle",
            "Any books published by Penguin Random House?",
            "Any mystery books?"
        ],
        title="📚 Library Assistant - Find Your Next Great Read",
        theme=gr.themes.Citrus()
    )


if __name__ == "__main__":
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    print("Processing Excel file...")
    documents = process_excel_rows_to_documents("src/library_agent/data/lib_books.xlsx")

    print("Running ingestion pipeline...")
    ingestion_pipeline = create_ingestion_pipeline(document_store)
    ingestion_pipeline.run({"cleaner": {"documents": documents}})

    demo = create_chat_agent_demo(document_store)
    demo.launch()

    