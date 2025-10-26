#!/usr/bin/env python3
"""
Index existing session data into ChromaDB
This script rebuilds the ChromaDB index from existing JSONL session files
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings

def get_file_hash(content: str) -> str:
    """Generate a hash for content to use as ID"""
    return hashlib.md5(content.encode()).hexdigest()

def index_sessions(data_dir: Path, chroma_path: Path):
    """Index all session files into ChromaDB"""
    
    print(f"Connecting to ChromaDB at {chroma_path}")
    
    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Get or create collections
    try:
        memories_collection = client.get_or_create_collection(
            name="memories",
            metadata={"description": "Memory fragments from conversations"}
        )
        
        sessions_collection = client.get_or_create_collection(
            name="sessions",
            metadata={"description": "Chat session metadata"}
        )
        
        print(f"Collections ready. Current counts:")
        print(f"  - memories: {memories_collection.count()} documents")
        print(f"  - sessions: {sessions_collection.count()} documents")
        
    except Exception as e:
        print(f"Error creating collections: {e}")
        return
    
    # Find all shard directories
    shards_dir = data_dir / "shards"
    if not shards_dir.exists():
        print(f"No shards directory found at {shards_dir}")
        return
    
    indexed_count = 0
    memory_count = 0
    
    # Iterate through all shard directories
    for shard_dir in shards_dir.iterdir():
        if not shard_dir.is_dir():
            continue
            
        shard_id = shard_dir.name
        print(f"\nProcessing shard: {shard_id}")
        
        # Sessions are directly in shard/sessions
        sessions_dir = shard_dir / "sessions"
        
        if not sessions_dir.exists():
            continue
        
        # Process each session file
        for session_file in sessions_dir.glob("*.jsonl"):
            try:
                session_id = session_file.stem
                
                # Extract user_id from filename
                # Format is typically: user_shard_session.jsonl
                parts = session_id.split('_')
                if len(parts) >= 2:
                    user_id = parts[0]
                else:
                    user_id = "unknown"
                
                # Check if already indexed
                existing = sessions_collection.get(
                    ids=[session_id],
                    include=[]
                )
                
                if existing['ids']:
                    print(f"  Session {session_id} already indexed, skipping")
                    continue
                
                # Read session file
                messages = []
                with open(session_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                msg = json.loads(line)
                                messages.append(msg)
                            except json.JSONDecodeError:
                                continue
                
                if not messages:
                    continue
                
                # Extract session metadata
                first_msg = messages[0]
                last_msg = messages[-1]
                
                # Index session metadata
                session_text = f"Session in {shard_id} for {user_id}"
                if messages:
                    # Add first user message as context
                    user_msgs = [m for m in messages if m.get('sender') == 'user']
                    if user_msgs:
                        session_text += f": {user_msgs[0].get('content', '')[:200]}"
                
                sessions_collection.add(
                    ids=[session_id],
                    documents=[session_text],
                    metadatas=[{
                        "shard_id": shard_id,
                        "user_id": user_id,
                        "message_count": len(messages),
                        "created_at": first_msg.get('timestamp', ''),
                        "updated_at": last_msg.get('timestamp', ''),
                        "file_path": str(session_file)
                    }]
                )
                indexed_count += 1
                
                # Extract and index memory fragments from assistant messages
                for msg in messages:
                    if msg.get('sender') in ['assistant', shard_id]:
                        content = msg.get('content', '')
                        if len(content) > 50:  # Only index substantial content
                            memory_id = get_file_hash(f"{session_id}_{msg.get('id', '')}")
                            
                            # Check if already exists
                            existing_memory = memories_collection.get(
                                ids=[memory_id],
                                include=[]
                            )
                            
                            if not existing_memory['ids']:
                                memories_collection.add(
                                    ids=[memory_id],
                                    documents=[content],
                                    metadatas=[{
                                        "session_id": session_id,
                                        "shard_id": shard_id,
                                        "user_id": user_id,
                                        "message_id": msg.get('id', ''),
                                        "timestamp": msg.get('timestamp', ''),
                                        "sender": msg.get('sender', 'assistant')
                                    }]
                                )
                                memory_count += 1
                
                print(f"  Indexed session {session_id} with {len(messages)} messages")
                
            except Exception as e:
                print(f"  Error processing {session_file}: {e}")
                continue
    
    print(f"\n=== Indexing Complete ===")
    print(f"Indexed {indexed_count} sessions")
    print(f"Created {memory_count} memory fragments")
    print(f"Final collection counts:")
    print(f"  - memories: {memories_collection.count()} documents")
    print(f"  - sessions: {sessions_collection.count()} documents")

def main():
    # Set paths
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data"
    chroma_path = data_dir / "chromadb_server"
    
    print("=== RoampalAI Data Indexer ===")
    print(f"Data directory: {data_dir}")
    print(f"ChromaDB path: {chroma_path}")
    
    # Ensure ChromaDB directory exists
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    # Run indexing
    index_sessions(data_dir, chroma_path)
    
    print("\nIndexing complete! ChromaDB is now populated with existing data.")
    print("You can start the ChromaDB server with:")
    print("  chroma run --path ./data/chromadb_server --port 8003")

if __name__ == "__main__":
    main()