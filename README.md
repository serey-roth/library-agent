# 📚 Seattle Public Library AI Assistant

An agentic Retrieval-Augmented Generation (RAG) assistant that helps users explore the Seattle Public Library catalog using natural language queries. Ask about books by topic, author, branch, or genre, and receive smart, AI-powered recommendations.

Built with Streamlit, Neo4j, and LangChain.

⚠️ **EXPERIMENTAL DEMO** - This demo project is not affiliated with Seattle Public Library.

## ✨ Features

- 🔎 **Smart Book Search** – Vector-based semantic search through book descriptions
- 👤 **Author Search** – Discover books by specific authors
- 📚 **Genre Recommendations** – Get suggestions based on categories and subjects
- 💬 **Conversational Interface** – Chat naturally with the AI assistant

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Database**: Neo4j (graph database + vector store)
- **AI/ML**: LangChain, LangGraph, OpenAI

### Data Sources

- [Seattle Library Collection Inventory](https://data.seattle.gov/Community-and-Culture/Library-Collection-Inventory/6vkj-f5xf/about_data)
- Google Books API

## 📂 Project Structure

```bash
src/
├── rag/
│   ├── main.py          # Streamlit entry point
│   ├── ingestion.py     # Load and index data into Neo4j
│   ├── tools.py         # Search + retrieval tools for the AI agent
│   └── prompts.py       # System + agent prompts
├── data/
│   └── lib_books.xlsx   # Library book dataset
└── logger.py            # Logging utilities
```
