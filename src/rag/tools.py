
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_community.vectorstores.neo4j_vector import Neo4jVector    
from langchain_neo4j import Neo4jGraph as LangchainNeo4jGraph
from langchain_openai import OpenAIEmbeddings


class SearchBooksArgs(BaseModel):
    query: Annotated[str, Field(description="The main query for semantic search (e.g., 'gothic books', 'mystery novels', 'science fiction')")]
    title: Annotated[str | None, Field(description="Book title to search for (e.g., 'The Great Gatsby')")]
    description: Annotated[str | None, Field(description="Book description to search for (e.g., 'A novel about a young man who inherits a large fortune and uses it to pursue a life of excess and excess')")]
    author: Annotated[str | None, Field(description="Author name to search for (e.g., 'Stephen King', 'Jane Austen')")]
    category: Annotated[str | None, Field(description="Book genre or category (e.g., 'gothic', 'mystery', 'science fiction', 'romance', 'thriller')")]
    publisher: Annotated[str | None, Field(description="Publisher name (e.g., 'Penguin', 'Random House')")]
    location: Annotated[str | None, Field(description="Library branch or location (e.g., 'Queen Anne', 'Downtown', 'Central')")]
    limit: Annotated[int, Field(description="Maximum number of results to return", default=10)]
    

def create_search_books_tool(graph: LangchainNeo4jGraph, vectorstore: Neo4jVector, embeddings: OpenAIEmbeddings) -> tool:
    def find_location_codes(location: str) -> list[str]:
        all_locations = graph.query("MATCH (l:Location) RETURN l.code as code, l.description as description")
        location_words = location.lower().split()
        results = [loc['code'] for loc in all_locations if any(word in loc['description'].lower() for word in location_words)]
        return results
    
    
    def build_cypher_query(query: str, title: str | None, description: str | None, author: str | None, category: str | None, publisher: str | None, location: str | None, limit: int) -> str:
        if location:
            location_codes = find_location_codes(location)
        else:
            location_codes = []
            
        where_conditions = []
        if title:
            where_conditions.append("lower(b.title) CONTAINS $title")
        if author:
            where_conditions.append("lower(a.name) CONTAINS $author")
        if category:
            where_conditions.append("lower(c.name) CONTAINS $category")
        if publisher:
            where_conditions.append("lower(p.name) CONTAINS $publisher")
        if len(location_codes) > 0:
            where_conditions.append("lower(l.code) IN $location_codes") 
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        cypher_match = "MATCH (b:Book)"
        if author:
            cypher_match += "\nMATCH (b)-[:WRITTEN_BY]->(a:Author)"
        if category:
            cypher_match += "\nMATCH (b)-[:BELONGS_TO]->(c:Category)"
        if publisher:
            cypher_match += "\nMATCH (b)-[:PUBLISHED_BY]->(p:Publisher)"
        if len(location_codes) > 0:
            cypher_match += "\nMATCH (b)-[:LOCATED_IN]->(l:Location)"
            
        cypher_return = f"RETURN DISTINCT b.title as title, b.subtitle as subtitle, b.isbn as isbn, b.published_date as published_date, b.norm_desc as description, b.authors as authors, b.categories as categories, b.publisher as publisher, b.locations as locations"
        
        order_by = "ORDER BY"
        if description:
            order_by += " vector.similarity.cosine(b.embedding, $embedded_query) DESC"
        else:
            order_by += " b.title ASC"
            
        cypher_query = f"""
            {cypher_match}
            WHERE {where_clause}
            {cypher_return}
            {order_by}
            LIMIT $limit
        """

        return cypher_query
    
    @tool(name_or_callable="search_books", description="Search for books using semantic vector search with optional filters.", args_schema=SearchBooksArgs)
    def search_books(
        query: Annotated[str, Field(description="The main query for semantic search (e.g., 'gothic books', 'mystery novels', 'science fiction')")],
        title: Annotated[str | None, Field(description="Book title to search for (e.g., 'The Great Gatsby')")],
        description: Annotated[str | None, Field(description="Book description to search for (e.g., 'A novel about a young man who inherits a large fortune and uses it to pursue a life of excess and excess')")],
        author: Annotated[str | None, Field(description="Author name to search for (e.g., 'Stephen King', 'Jane Austen')")],
        category: Annotated[str | None, Field(description="Book genre or category (e.g., 'gothic', 'mystery', 'science fiction', 'romance', 'thriller')")],
        publisher: Annotated[str | None, Field(description="Publisher name (e.g., 'Penguin', 'Random House')")],
        location: Annotated[str | None, Field(description="Library branch or location (e.g., 'Queen Anne', 'Downtown', 'Central')")],
        limit: Annotated[int, Field(description="Maximum number of results to return", default=10)]
    ) -> list[dict]: 
        if not title and not author and not description and not category and not publisher and not location:
            return vectorstore.similarity_search(query, k=limit)
        
        cypher_query = build_cypher_query(query, title, description, author, category, publisher, location, limit)
        
        params = {"limit": limit, "embedded_query": embeddings.embed_query(query)}
        if title:
            params["title"] = title.lower()
        if author:
            params["author"] = author.lower()
        if category:
            params["category"] = category.lower()
        if publisher:
            params["publisher"] = publisher.lower()
        if location:
            location_codes = find_location_codes(location)
            if location_codes:
                params["location_codes"] = [code.lower() for code in location_codes]
        
        return graph.query(cypher_query, params=params)
    
    return search_books