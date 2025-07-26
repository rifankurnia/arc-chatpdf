from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(TypedDict):
    """State management for the conversation flow"""
    messages: List[BaseMessage]
    intent_type: str
    query: str
    context: str
    search_results: List[Dict[str, Any]]
    retrieved_docs: List[Document]
    needs_clarification: bool
    final_answer: str


# ============================================================================
# UTILITY FUNCTIONS FOR MESSAGE HANDLING
# ============================================================================

def extract_text_from_message(message: Any) -> str:
    """Extract text content from various message formats."""
    # If it's already a string, return it
    if isinstance(message, str):
        return message
    
    # Handle LangChain message objects
    if hasattr(message, 'content'):
        content = message.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and content:
            # Handle nested content structure
            for item in content:
                if isinstance(item, dict) and item.get('text'):
                    return item['text']
                elif hasattr(item, 'text'):
                    return item.text
    
    # Handle dict format messages
    if isinstance(message, dict):
        if message.get('type') == 'human':
            content = message.get('content', [])
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and content:
                for item in content:
                    if isinstance(item, dict) and item.get('text'):
                        return item['text']
    
    return ""


def extract_query_from_state(state: Dict[str, Any]) -> str:
    """Extract the current query from the state."""
    # Extract from messages first (prioritize latest message)
    messages = state.get('messages', [])
    if messages:
        # Look through messages in reverse order
        for msg in reversed(messages):
            # Check if it's a human message
            is_human = False
            
            if hasattr(msg, 'type'):
                is_human = msg.type == 'human'
            elif isinstance(msg, dict) and msg.get('type') == 'human':
                is_human = True
            elif isinstance(msg, HumanMessage):
                is_human = True
            
            if is_human:
                text = extract_text_from_message(msg)
                if text:
                    return text
    
    # Fallback to direct query field if no human messages found
    if state.get('query'):
        return state['query']
    
    return ""


# ============================================================================
# STEP 1: ROUTING NODE (Entry Point)
# ============================================================================

def routing_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Entry point that analyzes the query and determines the processing path.
    
    Flow:
    1. Extract the user query
    2. Analyze if it's ambiguous
    3. Determine if it needs document search or web search
    4. Set routing flags
    """
    logger.info("=== ROUTING NODE ===")
    
    # Extract the query using our robust extraction method
    query = extract_query_from_state(state)
    
    if not query:
        logger.warning("No query found in state")
        return {
            "intent_type": "document_rag",
            "needs_clarification": True
        }
    
    logger.info(f"Query: {query}")
    
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant that determines how to handle user queries.
        
        Analyze the user's query and determine:
        1. If it's ambiguous and needs clarification
        2. If it can be answered from academic papers (document_rag)
        3. If it needs web search (websearch_rag)
        
        Respond with a JSON object containing:
        - intent_type: "document_rag" or "websearch_rag"
        - needs_clarification: true or false
        - reasoning: brief explanation
        
    
        """),
        ("human", "{query}")
    ])
    
    response = llm.invoke(routing_prompt.format_messages(query=query))
    
    # Parse the response
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            routing_info = json.loads(json_match.group())
        else:
            routing_info = {"intent_type": "document_rag", "needs_clarification": False}
    except Exception as e:
        logger.error(f"Error parsing routing response: {e}")
        routing_info = {"intent_type": "document_rag", "needs_clarification": False}
    
    # Check for explicit web search requests
    web_search_keywords = ["search online", "search the web", "current", "latest", 
                          "this month", "today", "recent news", "what's new"]
    if any(phrase in query.lower() for phrase in web_search_keywords):
        routing_info["intent_type"] = "websearch_rag"
    
    logger.info(f"Routing decision: {routing_info}")
    
    return {
        "intent_type": routing_info.get("intent_type", "document_rag"),
        "needs_clarification": routing_info.get("needs_clarification", False)
    }


def routing_conditional(state: State, config: RunnableConfig) -> Literal["document_rag", "websearch_rag", "clarification_node"]:
    """
    Conditional routing logic after routing_node.
    Determines which path to take based on the routing analysis.
    """
    # If clarification is needed, go directly to clarification
    if state.get("needs_clarification", False):
        logger.info("Routing to: clarification_node (needs clarification)")
        return "clarification_node"
    
    # Otherwise, route based on intent type
    intent = state.get("intent_type", "document_rag")
    logger.info(f"Routing to: {intent}")
    
    if intent == "websearch_rag":
        return "websearch_rag"
    else:
        return "document_rag"


# ============================================================================
# STEP 2A: DOCUMENT RAG NODE (Document Search Path)
# ============================================================================

def document_rag(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handles document-based queries using the vector store.
    
    Flow:
    1. First call: Prepares for document retrieval
    2. Second call (after retrieval): Processes retrieved documents
    """
    logger.info("=== DOCUMENT RAG NODE ===")
    
    # Check if we have already retrieved documents
    if state.get("retrieved_docs"):
        logger.info(f"Processing {len(state['retrieved_docs'])} retrieved documents...")
        
        # Process retrieved documents to create final answer
        doc_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing academic papers. 
            Extract and summarize the most relevant information from these documents to answer the user's query.
            Be specific and cite the source documents when possible.
            
            User Query: {query}
            
            Documents:
            {documents}
            
            Provide a comprehensive answer that directly addresses the query. If the documents don't contain 
            relevant information, clearly state that."""),
            ("human", "{query}")
        ])
        
        # Format documents for analysis
        docs_text = "\n\n---Document---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(state["retrieved_docs"][:5])
        ])
        
        if not docs_text.strip():
            return {
                "context": "No relevant documents found for your query.",
                "final_answer": "I couldn't find any relevant information in the available documents to answer your query. Could you please rephrase or provide more specific details?"
            }
        
        # Extract query for the prompt
        query = extract_query_from_state(state)
        
        response = llm.invoke(doc_analysis_prompt.format_messages(
            query=query,
            documents=docs_text
        ))
        
        # Mark as processed by setting final_answer
        return {
            "context": response.content,
            "final_answer": response.content,
            "retrieved_docs": state["retrieved_docs"]  # Keep the docs in state
        }
    
    # First call - prepare for document retrieval
    logger.info("Preparing for document retrieval...")
    return {
        "context": "Searching through academic papers for relevant information...",
        "retrieved_docs": []  # Initialize empty to trigger retrieval
    }


def chroma_conditional(state: State, config: RunnableConfig) -> Literal["chroma_retriever", "clarification_node"]:
    """
    Conditional routing for document RAG flow.
    Determines whether to retrieve documents or proceed to final answer.
    """
    # If we have a final answer, go to clarification
    if state.get("final_answer"):
        logger.info("Final answer ready, going to clarification")
        return "clarification_node"
    
    # If we already have retrieved docs with content, go to clarification
    if state.get("retrieved_docs") and len(state.get("retrieved_docs", [])) > 0:
        logger.info(f"Already have {len(state['retrieved_docs'])} documents, going to clarification")
        return "clarification_node"
    
    # Otherwise, retrieve documents
    logger.info("Need to retrieve documents")
    return "chroma_retriever"


# ============================================================================
# STEP 2B: WEB SEARCH NODE (Web Search Path)
# ============================================================================

def websearch_rag(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handles web search queries for current information.
    
    Flow:
    1. First call: Prepares for web search
    2. Second call (after search): Processes search results
    """
    logger.info("=== WEB SEARCH RAG NODE ===")
    
    # Check if we have search results to process
    if state.get("search_results"):
        logger.info("Processing search results...")
        
        # Process search results to create final answer
        search_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing web search results.
            Extract and summarize the most relevant and recent information to answer the user's query.
            Focus on the most credible and recent sources.
            
            User Query: {query}
            
            Search Results:
            {results}
            
            Provide a comprehensive answer based on the search results. Include relevant dates and sources 
            when mentioning specific information."""),
            ("human", "{query}")
        ])
        
        # Format search results
        results_text = "\n\n---Result---\n\n".join([
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {result.get('content', result.get('snippet', 'N/A'))}"
            for i, result in enumerate(state.get("search_results", [])[:5])
        ])
        
        if not results_text.strip():
            return {
                "context": "No search results found.",
                "final_answer": "I couldn't find any relevant information from web search. Please try rephrasing your query or being more specific."
            }
        
        response = llm.invoke(search_analysis_prompt.format_messages(
            query=extract_query_from_state(state),
            results=results_text
        ))
        
        return {
            "context": response.content,
            "final_answer": response.content
        }
    
    # First call - prepare for web search
    logger.info("Preparing for web search...")
    return {
        "context": "Searching the web for current information..."
    }


def websearch_conditional(state: State, config: RunnableConfig) -> Literal["tavily_search", "clarification_node"]:
    """
    Conditional routing for web search flow.
    Determines whether to perform search or proceed to final answer.
    """
    # If we already have search results or final answer, go to clarification
    if state.get("search_results") or state.get("final_answer"):
        logger.info("Search results already available, going to clarification")
        return "clarification_node"
    
    # Otherwise, perform search
    logger.info("Need to perform web search")
    return "tavily_search"


# ============================================================================
# STEP 3: CLARIFICATION NODE (Final Step)
# ============================================================================

def clarification_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Final node that handles clarifications or provides the final answer.
    
    This node:
    1. Asks for clarification if the query was ambiguous
    2. Provides the final answer if processing is complete
    3. Handles edge cases where no good answer was found
    """
    logger.info("=== CLARIFICATION NODE ===")
    
    # If we have a final answer from previous nodes, return it
    if state.get("final_answer"):
        logger.info("Returning final answer")
        return {
            "messages": state["messages"] + [AIMessage(content=state["final_answer"])]
        }
    
    # If clarification is needed (from routing node)
    if state.get("needs_clarification", False):
        logger.info("Asking for clarification")
        
        clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that clarifies ambiguous questions.
            The user asked: {query}
            
            This question seems ambiguous or underspecified. Ask a clarifying question to better understand 
            what the user needs. Be specific about what information would help answer their question.
            
            For example:
            - If they ask about "accuracy", ask which dataset or model they're referring to
            - If they ask about "the best", ask what criteria they want to use
            - If they ask about "examples", ask for which task or context"""),
            ("human", "{query}")
        ])
        
        response = llm.invoke(clarification_prompt.format_messages(query=extract_query_from_state(state)))
        return {
            "messages": state["messages"] + [response],
            "needs_clarification": False
        }
    
    # Fallback: compile answer from any available context
    if state.get("context"):
        logger.info("Compiling answer from context")
        return {
            "messages": state["messages"] + [AIMessage(content=state["context"])]
        }
    
    # Default response if nothing else worked
    logger.info("No answer available, asking for more details")
    return {
        "messages": state["messages"] + [
            AIMessage(content="I need more information to answer your question. Could you please provide more details or rephrase your query?")
        ]
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def process_tool_results(state: State, tool_name: str, results: Any) -> Dict[str, Any]:
    """
    Process results from external tools (Tavily search or Chroma retriever).
    Converts tool outputs into state updates.
    """
    logger.info(f"Processing results from {tool_name}")
    
    if tool_name == "tavily_search":
        # Process Tavily search results
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except:
                results = [{"content": results, "title": "Search Result"}]
        
        # Ensure we have a list of results
        if isinstance(results, list):
            search_results = results
        elif isinstance(results, dict):
            # Handle case where Tavily returns a dict with 'results' key
            search_results = results.get('results', [results])
        else:
            search_results = [{"content": str(results), "title": "Search Result"}]
        
        logger.info(f"Processed {len(search_results)} search results")
        return {"search_results": search_results}
    
    elif tool_name == "chroma_retriever":
        # Process Chroma retrieval results
        if isinstance(results, list) and results:
            logger.info(f"Retrieved {len(results)} documents")
            # Make sure we're returning actual documents
            return {"retrieved_docs": results}
        else:
            logger.info("No documents retrieved")
            # Return empty list but not None
            return {"retrieved_docs": []}
    
    return {}