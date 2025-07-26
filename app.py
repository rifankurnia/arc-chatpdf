from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from main import graph, run_query
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chat With PDF API",
    description="An intelligent question-answering system for PDF documents with web search capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int

# In-memory session storage (in production, use Redis or similar)
sessions: Dict[str, Dict[str, Any]] = {}

# Helper functions
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create a new one"""
    if session_id and session_id in sessions:
        return session_id
    
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
        "message_count": 0
    }
    return new_session_id

def get_session_messages(session_id: str) -> List[BaseMessage]:
    """Retrieve messages for a session"""
    if session_id not in sessions:
        return []
    return sessions[session_id]["messages"]

def add_to_session(session_id: str, human_msg: str, ai_msg: str):
    """Add messages to session history"""
    if session_id in sessions:
        sessions[session_id]["messages"].extend([
            HumanMessage(content=human_msg),
            AIMessage(content=ai_msg)
        ])
        sessions[session_id]["message_count"] += 2

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chat With PDF API",
        "endpoints": {
            "POST /query": "Submit a question",
            "DELETE /session/{session_id}": "Clear a session",
            "GET /session/{session_id}": "Get session info",
            "GET /health": "Health check"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Submit a question to the system.
    
    The system will:
    1. Analyze the query to determine if it needs clarification
    2. Route to either document search or web search
    3. Return a comprehensive answer
    """
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Get session history
        message_history = get_session_messages(session_id)
        
        logger.info(f"Processing query: {request.query} for session: {session_id}")
        
        # Run the query through the graph
        response = await run_query(request.query, message_history)
        
        # Add to session history
        add_to_session(session_id, request.query, response)
        
        return QueryResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session's memory"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": f"Session {session_id} cleared successfully"}

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=session["message_count"]
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "sessions_active": len(sessions)
    }

# Optional: Endpoint to test individual components
@app.post("/debug/routing")
async def debug_routing(request: QueryRequest):
    """Debug endpoint to see how queries are routed"""
    from agents import routing_node, State
    
    state = State(
        messages=[HumanMessage(content=request.query)],
        query=request.query,
        intent_type="",
        context="",
        search_results=[],
        retrieved_docs=[],
        needs_clarification=False,
        final_answer=""
    )
    
    result = routing_node(state, {})
    return {
        "query": request.query,
        "routing_result": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)