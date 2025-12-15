#!/usr/bin/env python3
"""
Migration script to move knowledge graph data from old hardcoded path 
to new shard-based structure.
"""

import json
import shutil
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

logger = logging.getLogger(__name__)

def migrate_knowledge_graph_data():
    """Migrate knowledge graph data from old path to new shard-based structure"""
    
    # Old path structure
    old_graph_path = Path("data/og_data/og_data/knowledge_graph/knowledge_graph.json")
    
    # New path structure for 'og' shard
    new_graph_dir = settings.paths.get_knowledge_graph_path("og")
    new_graph_path = new_graph_dir / "knowledge_graph.json"
    
    print(f"🔍 Checking for existing knowledge graph data...")
    print(f"   Old path: {old_graph_path}")
    print(f"   New path: {new_graph_path}")
    
    # Check if old data exists
    if not old_graph_path.exists():
        print("✅ No old knowledge graph data found - nothing to migrate")
        return
    
    # Check if new data already exists
    if new_graph_path.exists():
        print("⚠️  New knowledge graph data already exists")
        print(f"   Old: {old_graph_path}")
        print(f"   New: {new_graph_path}")
        
        # Compare file sizes
        old_size = old_graph_path.stat().st_size
        new_size = new_graph_path.stat().st_size
        
        print(f"   Old file size: {old_size} bytes")
        print(f"   New file size: {new_size} bytes")
        
        if old_size > new_size:
            print("   Old file is larger - backing up new and copying old")
            # Backup new file
            backup_path = new_graph_path.with_suffix('.json.backup')
            shutil.copy2(new_graph_path, backup_path)
            print(f"   Backup created: {backup_path}")
        
        response = input("   Do you want to overwrite the new data? (y/N): ")
        if response.lower() != 'y':
            print("   Migration cancelled")
            return
    
    try:
        # Create new directory
        new_graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(old_graph_path, new_graph_path)
        
        # Verify the copy
        if new_graph_path.exists():
            print(f"✅ Successfully migrated knowledge graph data")
            print(f"   From: {old_graph_path}")
            print(f"   To: {new_graph_path}")
            
            # Load and validate the data
            try:
                with open(new_graph_path, 'r') as f:
                    data = json.load(f)
                
                # Update metadata to reflect new shard structure
                if 'metadata' in data:
                    data['metadata']['shard_id'] = 'og'
                    data['metadata']['migrated'] = True
                    
                    with open(new_graph_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                    
                    print("   ✅ Updated metadata to reflect new shard structure")
                
            except Exception as e:
                print(f"   ⚠️  Warning: Could not validate migrated data: {e}")
        
        # Optionally remove old file
        response = input("   Remove old knowledge graph file? (y/N): ")
        if response.lower() == 'y':
            old_graph_path.unlink()
            print(f"   ✅ Removed old file: {old_graph_path}")
        else:
            print(f"   ℹ️  Old file preserved: {old_graph_path}")
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return
    
    print("\n🎉 Knowledge graph migration completed successfully!")
    print("   The knowledge graph is now properly organized per shard.")
    print("   Only tone and user profile data persist across shards.")

if __name__ == "__main__":
    migrate_knowledge_graph_data() 