"""
Test script to simulate the full speaker detection flow as used by the app
"""
import requests
import time
import os
import json

BASE_URL = "http://localhost:5000"

def test_full_speaker_detection():
    """Test the complete speaker detection flow"""
    
    # 1. Upload the podcast file
    print("=" * 60)
    print("Step 1: Uploading podcast file...")
    print("=" * 60)
    
    podcast_path = r"c:\Users\sameer\OneDrive\4_Projects\Naarad\Product\samples\Ep1_full_without_music.wav"
    ad_script_path = r"c:\Users\sameer\OneDrive\4_Projects\Naarad\Product\samples\sample_ad_script.txt"
    
    if not os.path.exists(podcast_path):
        print(f"ERROR: Podcast file not found: {podcast_path}")
        return
    
    if not os.path.exists(ad_script_path):
        print(f"ERROR: Ad script file not found: {ad_script_path}")
        return
    
    with open(podcast_path, 'rb') as podcast_file, open(ad_script_path, 'rb') as ad_script_file:
        files = {
            'podcast': ('Ep1_full_without_music.wav', podcast_file, 'audio/wav'),
            'adScript': ('sample_ad_script.txt', ad_script_file, 'text/plain')
        }
        data = {'workflow': 'full'}
        
        response = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
    
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
    
    job_id = response.json().get('job_id')
    print(f"Upload successful. Job ID: {job_id}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # 2. Start speaker detection
    print("\n" + "=" * 60)
    print("Step 2: Starting speaker detection...")
    print("=" * 60)
    
    response = requests.post(f"{BASE_URL}/api/detect-speakers/{job_id}")
    print(f"Detection started. Response: {json.dumps(response.json(), indent=2)}")
    
    # 3. Poll for speaker status
    print("\n" + "=" * 60)
    print("Step 3: Polling for speaker status...")
    print("=" * 60)
    
    max_polls = 120  # 2 minutes max
    poll_count = 0
    
    while poll_count < max_polls:
        response = requests.get(f"{BASE_URL}/api/speaker-status/{job_id}")
        status_data = response.json()
        
        speakers_found = status_data.get('speakers_found', [])
        detection_complete = status_data.get('detection_complete', False)
        message = status_data.get('message', '')
        progress = status_data.get('progress', 0)
        
        print(f"[Poll {poll_count + 1}] Progress: {progress}% | Speakers: {len(speakers_found)} | Complete: {detection_complete} | {message}")
        
        # Show speaker details if found
        if speakers_found:
            for speaker in speakers_found:
                print(f"    Speaker {speaker['id']}: {speaker.get('total_speaking_time', 0):.1f}s of speech, sample: {speaker.get('sample_path', 'N/A')}")
        
        if detection_complete:
            print("\nSpeaker detection complete!")
            break
        
        time.sleep(1)
        poll_count += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/speaker-status/{job_id}")
    final_data = response.json()
    
    print(f"Total speakers found: {final_data.get('num_speakers', 0)}")
    print(f"Requires selection: {final_data.get('requires_selection', False)}")
    
    speakers = final_data.get('speakers_found', [])
    for speaker in speakers:
        print(f"  Speaker {speaker['id']}:")
        print(f"    - Speaking time: {speaker.get('total_speaking_time', 0):.1f}s")
        print(f"    - Sample path: {speaker.get('sample_path', 'N/A')}")
        # Try to access the sample
        sample_response = requests.get(f"{BASE_URL}/api/speaker-sample/{job_id}/{speaker['id']}")
        print(f"    - Sample accessible: {sample_response.status_code == 200}")

if __name__ == "__main__":
    test_full_speaker_detection()
