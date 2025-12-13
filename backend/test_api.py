"""
API Test script to test the full upload and process flow via HTTP
"""
import os
import sys
import time
import requests

BASE_URL = "http://localhost:5000"

def main():
    # File paths
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    podcast_path = os.path.join(samples_dir, 'Ep1_AClip1.wav')
    ad_script_path = os.path.join(samples_dir, 'sample_ad_script.txt')
    
    print(f"Podcast: {podcast_path}")
    print(f"Ad Script: {ad_script_path}")
    
    # Verify files exist
    if not os.path.exists(podcast_path):
        print(f"❌ Podcast file not found: {podcast_path}")
        return
    if not os.path.exists(ad_script_path):
        print(f"❌ Ad script file not found: {ad_script_path}")
        return
    
    print(f"✅ Input files exist")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return
    
    # Upload files
    print("\n2. Uploading files...")
    try:
        with open(podcast_path, 'rb') as podcast_file, open(ad_script_path, 'rb') as script_file:
            files = {
                'podcast': ('Ep1_AClip1.wav', podcast_file, 'audio/wav'),
                'adScript': ('sample_ad_script.txt', script_file, 'text/plain')
            }
            r = requests.post(f"{BASE_URL}/api/upload", files=files, timeout=30)
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
            
            if r.status_code != 200:
                print(f"❌ Upload failed")
                return
            
            job_id = r.json().get('job_id')
            print(f"   Job ID: {job_id}")
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start processing
    print("\n3. Starting processing...")
    try:
        r = requests.post(f"{BASE_URL}/api/process/{job_id}", timeout=10)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
        
        if r.status_code != 200:
            print(f"❌ Process start failed")
            return
    except Exception as e:
        print(f"❌ Process start error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Poll for status
    print("\n4. Polling for status...")
    max_wait = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            r = requests.get(f"{BASE_URL}/api/status/{job_id}", timeout=10)
            data = r.json()
            
            status = data.get('status')
            progress = data.get('progress', 0)
            message = data.get('message', '')
            
            print(f"   [{progress}%] {status}: {message}")
            
            if status == 'completed':
                print(f"\n✅ Processing completed!")
                print(f"   Output: {data.get('output_path')}")
                
                # Try to download
                print("\n5. Testing download...")
                r = requests.get(f"{BASE_URL}/api/download/{job_id}", timeout=30)
                print(f"   Status: {r.status_code}")
                if r.status_code == 200:
                    output_file = os.path.join(os.path.dirname(__file__), 'test_outputs', 'api_test_output.wav')
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'wb') as f:
                        f.write(r.content)
                    print(f"   ✅ Downloaded to: {output_file}")
                    print(f"   File size: {len(r.content)} bytes")
                return
            
            if status == 'error':
                print(f"\n❌ Processing failed!")
                print(f"   Error: {data.get('error')}")
                return
            
            time.sleep(3)
            
        except Exception as e:
            print(f"   Status check error: {e}")
            time.sleep(3)
    
    print(f"\n❌ Timeout waiting for processing to complete")

if __name__ == '__main__':
    main()
