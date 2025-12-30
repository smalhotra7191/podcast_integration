"""
Test script for voice cloning with XTTS-v2
Tests the updated voice cloning module with sample inputs
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.voice_cloning import VoiceCloningAgent

def test_voice_cloning(use_full_podcast=False):
    """Test voice cloning with sample inputs"""
    
    # Paths to sample files
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    # Use full podcast for better voice conditioning (more samples = better similarity)
    if use_full_podcast:
        speaker_sample = os.path.join(samples_dir, 'Ep1_full_without_music.wav')
    else:
        speaker_sample = os.path.join(samples_dir, 'Ep1_AClip1.wav')
    ad_script_path = os.path.join(samples_dir, 'sample_ad_script.txt')
    
    # Output path
    output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_cloned_ad.wav')
    
    # Check files exist
    print(f"Speaker sample: {speaker_sample}")
    print(f"  Exists: {os.path.exists(speaker_sample)}")
    
    print(f"Ad script: {ad_script_path}")
    print(f"  Exists: {os.path.exists(ad_script_path)}")
    
    if not os.path.exists(speaker_sample):
        print("ERROR: Speaker sample not found!")
        return
    
    if not os.path.exists(ad_script_path):
        print("ERROR: Ad script not found!")
        return
    
    # Read ad script
    with open(ad_script_path, 'r') as f:
        ad_script = f.read()
    
    # Use the full ad script for comprehensive testing
    test_script = ad_script
    
    print(f"\nTest script: {test_script[:100]}...")
    print(f"\nInitializing Voice Cloning Agent...")
    
    # Initialize agent
    agent = VoiceCloningAgent()
    print(f"Device: {agent.device}")
    
    # Generate cloned audio
    print(f"\nGenerating cloned audio...")
    result_path = agent.generate_ad_audio(
        ad_script=test_script,
        speaker_sample_path=speaker_sample,
        output_path=output_path
    )
    
    print(f"\nOutput saved to: {result_path}")
    print(f"File exists: {os.path.exists(result_path)}")
    
    # Get similarity score
    similarity = agent.get_similarity_score()
    print(f"\n{'='*50}")
    print(f"VOICE SIMILARITY RESULTS")
    print(f"{'='*50}")
    print(f"Score: {similarity['score']}%")
    print(f"Quality: {similarity['quality_label']}")
    print(f"Is Same Speaker: {similarity['is_same_speaker']}")
    print(f"Raw Score: {similarity['raw_score']}")
    print(f"{'='*50}")
    
    if similarity['score'] >= 80:
        print("\n[SUCCESS] Voice similarity is above 80%!")
    else:
        print(f"\n[WARNING] Voice similarity is {similarity['score']}%, below 80% target")
    
    return similarity

if __name__ == "__main__":
    test_voice_cloning()
