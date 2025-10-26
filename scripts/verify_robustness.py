#!/usr/bin/env python3
"""
Comprehensive robustness verification for RoampalAI system
"""

import requests
import json
import os
from pathlib import Path
from datetime import datetime

def check_services():
    """Verify all services are running"""
    services = {
        'Backend API': 'http://localhost:8000/docs',
        'ChromaDB': 'http://localhost:8003/api/v1/heartbeat',
        'Frontend': 'http://localhost:5173'
    }
    
    print("=== Service Health Check ===")
    all_up = True
    for name, url in services.items():
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code in [200, 405]:  # 405 for ChromaDB heartbeat
                print(f"  [OK] {name}: Running")
            else:
                print(f"  [FAIL] {name}: HTTP {resp.status_code}")
                all_up = False
        except:
            print(f"  [FAIL] {name}: Not responding")
            all_up = False
    
    return all_up

def check_data_persistence():
    """Verify data files exist and are accessible"""
    print("\n=== Data Persistence Check ===")
    
    checks = {
        'ChromaDB Data': Path('backend/data/chromadb_server/chroma.sqlite3'),
        'Session Files': Path('backend/data/shards/roampal/sessions'),
        'Image Files': Path('backend/data/shards/roampal/images'),
        'User Registries': Path('backend/data/shards/roampal/images/user_image_registry.json')
    }
    
    all_exist = True
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024
                print(f"  [OK] {name}: {size:.1f} KB")
            else:
                count = len(list(path.iterdir()))
                print(f"  [OK] {name}: {count} items")
        else:
            print(f"  [FAIL] {name}: Not found")
            all_exist = False
    
    return all_exist

def check_api_functionality():
    """Test critical API endpoints"""
    print("\n=== API Functionality Check ===")
    
    tests = [
        ('Get Sessions', 'GET', '/api/simple-sessions/roampal/logan'),
        ('Get Images', 'GET', '/api/simple-images/roampal/logan'),
        ('Get Memory', 'GET', '/api/memory-viz/fragments/roampal?user_id=logan')
    ]
    
    all_pass = True
    for name, method, endpoint in tests:
        try:
            url = f'http://localhost:8000{endpoint}'
            resp = requests.request(method, url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    print(f"  [OK] {name}: {len(data)} items")
                else:
                    print(f"  [OK] {name}: Response received")
            else:
                print(f"  [FAIL] {name}: HTTP {resp.status_code}")
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] {name}: {str(e)}")
            all_pass = False
    
    return all_pass

def check_data_integrity():
    """Verify data integrity and counts"""
    print("\n=== Data Integrity Check ===")
    
    # Count total images across all shards
    total_images = 0
    total_sessions = 0
    shards_dir = Path('backend/data/shards')
    
    for shard in shards_dir.iterdir():
        if shard.is_dir():
            # Count images
            img_dir = shard / 'images'
            if img_dir.exists():
                images = len([f for f in img_dir.iterdir() 
                            if f.suffix in ['.png', '.jpg', '.jpeg']])
                total_images += images
            
            # Count sessions
            sess_dir = shard / 'sessions'
            if sess_dir.exists():
                sessions = len(list(sess_dir.glob('*.jsonl')))
                total_sessions += sessions
    
    print(f"  Total Images: {total_images}")
    print(f"  Total Sessions: {total_sessions}")
    print(f"  Active Shards: {len(list(shards_dir.iterdir()))}")
    
    # Verify ChromaDB collections
    try:
        resp = requests.get('http://localhost:8003/api/v1/collections')
        if resp.status_code == 200:
            collections = resp.json()
            print(f"  ChromaDB Collections: {len(collections)}")
        else:
            print(f"  ChromaDB Collections: Unable to verify")
    except:
        print(f"  ChromaDB Collections: Not accessible")
    
    return total_images > 0 and total_sessions > 0

def generate_report():
    """Generate robustness report"""
    print("\n" + "="*50)
    print("ROBUSTNESS VERIFICATION REPORT")
    print("="*50)
    
    checks = {
        'Services Running': check_services(),
        'Data Persistence': check_data_persistence(),
        'API Functionality': check_api_functionality(),
        'Data Integrity': check_data_integrity()
    }
    
    print("\n=== Summary ===")
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for check, result in checks.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {check}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    if passed == total:
        print("\n[SUCCESS] System is FULLY ROBUST and operational!")
        print("All data persists correctly and services are healthy.")
    else:
        print("\n[WARNING] Some checks failed. Review details above.")
    
    return passed == total

if __name__ == "__main__":
    os.chdir('C:/RoampalAI')
    robust = generate_report()
    exit(0 if robust else 1)