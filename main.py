from langgraph.graph import StateGraph, START, END
from agents import (
    State,
    clarification_node,
    routing_node,
    document_rag,
    websearch_rag,
    routing_conditional,
    websearch_conditional,
    chroma_conditional,
    process_tool_results
)
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage
import os
from typing import Dict, Any, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure LangSmith tracing
import os
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "arc-chatpdf")


# Custom tool wrappers to process results
def tavily_search_wrapper(query: str) -> Dict[str, Any]:
    """Wrapper for Tavily search that returns structured results"""
    tavily = TavilySearch(max_results=5)
    results = tavily.invoke({"query": query})
    return results


def remove_duplicate_docs(docs: List) -> List:
    """Remove duplicate documents based on content similarity"""
    seen_contents = set()
    unique_docs = []
    
    for doc in docs:
        # Create a content hash (first 100 chars)
        content_hash = doc.page_content[:100].strip()
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs


def chroma_retriever_wrapper(query: str) -> Dict[str, Any]:
    """Wrapper for Chroma retriever that returns documents"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    
    # Simple retrieval without query expansion
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 15  # Increased from 5 to 15
        }
    )
    docs = retriever.invoke(query)
    
    # Remove duplicates and sort by relevance
    unique_docs = remove_duplicate_docs(docs)
    return unique_docs[:20]  # Return top 20 unique documents


# Custom nodes that integrate with tools
def tavily_node(state: State) -> Dict[str, Any]:
    """Custom Tavily search node"""
    from agents import extract_query_from_state
    query = extract_query_from_state(state)
    if query:
        results = tavily_search_wrapper(query)
        return process_tool_results(state, "tavily_search", results)
    return {}


def chroma_node(state: State) -> Dict[str, Any]:
    """Custom Chroma retrieval node"""
    from agents import extract_query_from_state
    # Always extract the latest query from messages
    query = extract_query_from_state(state)
    if query:
        docs = chroma_retriever_wrapper(query)
        return process_tool_results(state, "chroma_retriever", docs)
    return {}


# Initialize the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("clarification_node", clarification_node)
graph_builder.add_node("routing_node", routing_node)
graph_builder.add_node("document_rag", document_rag)
graph_builder.add_node("websearch_rag", websearch_rag)
graph_builder.add_node("tavily_search", tavily_node)
graph_builder.add_node("chroma_retriever", chroma_node)

# Define the flow
graph_builder.add_edge(START, "routing_node")

# Routing node decides the path
graph_builder.add_conditional_edges(
    "routing_node",
    routing_conditional,
    {
        "document_rag": "document_rag",
        "websearch_rag": "websearch_rag",
        "clarification_node": "clarification_node"
    }
)

# Document RAG flow
graph_builder.add_conditional_edges(
    "document_rag",
    chroma_conditional,
    {
        "chroma_retriever": "chroma_retriever",
        "end": END
    }
)
graph_builder.add_edge("chroma_retriever", "document_rag")

# Web search flow
graph_builder.add_conditional_edges(
    "websearch_rag",
    websearch_conditional,
    {
        "tavily_search": "tavily_search",
        "end": END
    }
)
graph_builder.add_edge("tavily_search", "websearch_rag")

# Clarification node only for ambiguous queries
graph_builder.add_edge("clarification_node", END)

# Compile the graph
graph = graph_builder.compile()


# Helper function to run a query
async def run_query(query: str, message_history: list = None) -> str:
    """Execute a query through the graph"""
    if message_history is None:
        message_history = []
    
    # Add the current query to message history
    from langchain_core.messages import HumanMessage
    message_history.append(HumanMessage(content=query))
    
    # Initialize state
    initial_state = {
        "messages": message_history,
        "query": query,
        "intent_type": "",
        "context": "",
        "search_results": [],
        "retrieved_docs": [],
        "needs_clarification": False,
        "final_answer": ""
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state)
    
    # Extract the final answer from messages or final_answer field
    if result.get("final_answer"):
        # Add the final answer to message history for future context
        from langchain_core.messages import AIMessage
        message_history.append(AIMessage(content=result["final_answer"]))
        return result["final_answer"]
    elif result.get("messages") and len(result["messages"]) > len(message_history):
        # Return the last AI message and update message history
        last_message = result["messages"][-1]
        message_history.append(last_message)
        return last_message.content
    else:
        return "No response generated."


# Synchronous wrapper for convenience
def query_sync(query: str, message_history: list = None) -> str:
    """Synchronous version of run_query"""
    import asyncio
    return asyncio.run(run_query(query, message_history))