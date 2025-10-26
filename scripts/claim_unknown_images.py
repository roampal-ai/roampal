#!/usr/bin/env python3
"""
Allow a user to claim unknown images as their own
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def reassign_images_to_user(shard: str, target_user: str):
    """Reassign all unknown images in a shard to a specific user"""
    
    backend_dir = Path("C:/RoampalAI/backend")
    registry_file = backend_dir / "data" / "shards" / shard / "images" / "user_image_registry.json"
    
    if not registry_file.exists():
        print(f"No registry found for shard {shard}")
        return 0
    
    # Load registry
    with open(registry_file, 'r') as f:
        registry = json.load(f)
    
    # Get unknown images
    unknown_images = registry['users'].get('unknown', {}).get('images', [])
    if not unknown_images:
        print(f"No unknown images in {shard}")
        return 0
    
    print(f"Found {len(unknown_images)} unknown images in {shard}")
    
    # Move images to target user
    if target_user not in registry['users']:
        registry['users'][target_user] = {
            'image_count': 0,
            'total_size': 0,
            'images': []
        }
    
    # Add unknown images to target user
    for img in unknown_images:
        img['user'] = target_user
        registry['users'][target_user]['images'].append(img)
    
    # Update counts
    registry['users'][target_user]['image_count'] += len(unknown_images)
    registry['users'][target_user]['total_size'] += sum(img['size'] for img in unknown_images)
    
    # Remove from unknown
    if 'unknown' in registry['users']:
        registry['users']['unknown']['images'] = []
        registry['users']['unknown']['image_count'] = 0
        registry['users']['unknown']['total_size'] = 0
    
    # Update registry metadata
    registry['updated_at'] = datetime.now().isoformat()
    
    # Save updated registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully reassigned {len(unknown_images)} images to {target_user}")
    return len(unknown_images)

def main():
    print("=== Claim Unknown Images ===\n")
    
    # Reassign all unknown images to logan
    shards = ['roampal', 'service', 'stress_test_shard', 'test_shard_1', 'test_shard_2', 'test_shard_3']
    total_reassigned = 0
    
    for shard in shards:
        count = reassign_images_to_user(shard, 'logan')
        total_reassigned += count
    
    print(f"\n=== Complete ===")
    print(f"Total images reassigned to logan: {total_reassigned}")
    print("\nThe image gallery will now show all these images under logan's account")

if __name__ == "__main__":
    main()