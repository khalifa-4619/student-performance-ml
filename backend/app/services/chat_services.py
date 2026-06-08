from backend.app.core import config

def process_chat_session(session_id: str, user_utterance: str) -> dict:
    """
    Orchestrates state validation, system instruction initialization, 
    and context compilation across the pluggable abstract storage layer.
    """
    # Fetch current session thread history
    history_thread = config.memory_store.get(session_id)
    
    # SYSTEM INITIALIZATION CONTEXT: If history is empty, explicitly set up system rules first
    if not history_thread:
        system_instruction = {
            "role": "system", 
            "content": "You are the Early Warning Platform Assistant advising on student risk indicators."
        }
        config.memory_store.set(session_id, system_instruction)
        
    # Append the new incoming user message to the repository
    user_message_frame = {"role": "user", "content": user_utterance}
    config.memory_store.set(session_id, user_message_frame)
    
    # Read the updated history back to compute context frames accurately
    updated_history = config.memory_store.get(session_id)
    context_depth = len(updated_history)
    
    # Rule engine simulation (to be swapped out for an LLM later)
    response_content = (
        f"Understood. Service-layer worker analyzed student metrics. "
        f"The isolated contract layer safely preserves {context_depth} active context frames."
    )
    
    assistant_message_frame = {"role": "assistant", "content": response_content}
    config.memory_store.set(session_id, assistant_message_frame)
    
    return {
        "session_id": session_id,
        "history": config.memory_store.get(session_id),
        "status": "synchronized"
    }