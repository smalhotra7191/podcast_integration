"""
Test script to verify speaker identification is working correctly.
"""
import os
import sys
import tempfile

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.podcast_analyzer import PodcastAnalyzer

def test_speaker_identification():
    """Test the speaker identification functionality"""
    
    print("=" * 60)
    print("Testing Speaker Identification")
    print("=" * 60)
    
    # Initialize the analyzer
    print("\n1. Initializing PodcastAnalyzer...")
    analyzer = PodcastAnalyzer(speed_mode='fast')
    print("   ✓ PodcastAnalyzer initialized")
    
    # Check for test audio files
    test_audio_paths = []
    
    # Look in uploads folder
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if f.endswith(('.mp3', '.wav', '.m4a')):
                test_audio_paths.append(os.path.join(uploads_dir, f))
    
    # Look in samples folder
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.endswith(('.mp3', '.wav', '.m4a')):
                test_audio_paths.append(os.path.join(samples_dir, f))
    
    if not test_audio_paths:
        print("\n⚠ No test audio files found in 'uploads' or 'samples' directories.")
        print("  Please provide a path to an audio file:")
        audio_path = input("  Audio path: ").strip()
        if audio_path:
            test_audio_paths = [audio_path]
        else:
            print("  No audio file provided. Exiting.")
            return
    
    print(f"\n2. Found {len(test_audio_paths)} audio file(s) to test:")
    for path in test_audio_paths[:5]:  # Show first 5
        print(f"   - {os.path.basename(path)}")
    
    # Test with first available audio file
    audio_path = test_audio_paths[0]
    print(f"\n3. Testing speaker detection with: {os.path.basename(audio_path)}")
    
    # Create a temp output folder for speaker samples
    output_folder = tempfile.mkdtemp(prefix='speaker_test_')
    print(f"   Output folder: {output_folder}")
    
    def status_callback(progress, message):
        print(f"   [{progress:3d}%] {message}")
    
    try:
        print("\n4. Running speaker detection...")
        result = analyzer.detect_speakers(audio_path, output_folder, status_callback)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\n   Number of speakers detected: {result['num_speakers']}")
        print(f"   Requires user selection: {result['requires_selection']}")
        print(f"   Audio duration: {result['duration']:.1f}s")
        
        print("\n   Speakers found:")
        for speaker in result['speakers']:
            print(f"\n   Speaker {speaker['id'] + 1} ('{speaker['label']}'):")
            print(f"      - Speaking time: {speaker['total_speaking_time']:.1f}s")
            print(f"      - Sample file: {os.path.basename(speaker['sample_path'])}")
            print(f"      - Sample exists: {os.path.exists(speaker['sample_path'])}")
            print(f"      - Number of segments: {len(speaker.get('segments', []))}")
        
        if result['num_speakers'] > 1:
            print("\n   ✓ SUCCESS: Multiple speakers detected!")
            print("     The speaker selection feature should work in the UI.")
        else:
            print("\n   ℹ Note: Only 1 speaker detected.")
            print("     This could mean:")
            print("     - The audio actually has only one speaker")
            print("     - The speakers have similar voices")
            print("     - Try with a different audio file that has clearly distinct voices")
        
        # Test the speaker embedding model directly
        print("\n5. Testing speaker embedding model...")
        try:
            analyzer._load_speaker_model()
            print("   ✓ Speaker model loaded successfully")
            
            # Test embedding extraction
            if result['speakers']:
                sample_path = result['speakers'][0]['sample_path']
                embedding = analyzer._get_speaker_embedding(sample_path)
                print(f"   ✓ Speaker embedding shape: {embedding.shape}")
                print(f"   ✓ Embedding extraction working!")
        except Exception as e:
            print(f"   ✗ Error loading speaker model: {e}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print(f"\n   Note: Test outputs saved in: {output_folder}")
        print("   (You can delete this folder manually)")

if __name__ == "__main__":
    test_speaker_identification()
