"""
Test script to validate the full podcast ad integration pipeline
"""
import os
import sys
import traceback

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step(step_name, func):
    """Helper to test a step with error handling"""
    print(f"\n{'='*60}")
    print(f"Testing: {step_name}")
    print('='*60)
    try:
        result = func()
        print(f"✅ {step_name} - SUCCESS")
        return result
    except Exception as e:
        print(f"❌ {step_name} - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def main():
    # File paths
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    podcast_path = os.path.join(samples_dir, 'Ep1_AClip1.wav')
    ad_script_path = os.path.join(samples_dir, 'sample_ad_script.txt')
    output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Podcast: {podcast_path}")
    print(f"Ad Script: {ad_script_path}")
    print(f"Output Dir: {output_dir}")
    
    # Verify files exist
    if not os.path.exists(podcast_path):
        print(f"❌ Podcast file not found: {podcast_path}")
        return
    if not os.path.exists(ad_script_path):
        print(f"❌ Ad script file not found: {ad_script_path}")
        return
    
    print(f"✅ Input files exist")
    
    # Step 1: Test Ad Analyzer
    ad_analysis = test_step("Ad Script Analysis", lambda: test_ad_analyzer(ad_script_path))
    if ad_analysis is None:
        return
    
    # Step 2: Test Podcast Analyzer - Transcription
    transcription = test_step("Podcast Transcription", lambda: test_transcription(podcast_path))
    if transcription is None:
        return
    
    # Step 3: Test Content Analysis
    content_analysis = test_step("Content Analysis", lambda: test_content_analysis(transcription, ad_analysis))
    if content_analysis is None:
        return
    
    # Step 4: Test Speaker Sample Extraction
    speaker_sample = test_step("Speaker Sample Extraction", lambda: test_speaker_sample(podcast_path))
    if speaker_sample is None:
        return
    
    # Step 5: Test Voice Cloning
    ad_script = "Speaking of managing your life better, let me tell you about BudgetPro. It's an amazing app that helps you take control of your finances. Now back to what we were discussing."
    ad_audio_path = os.path.join(output_dir, 'test_ad.wav')
    ad_audio = test_step("Voice Cloning / TTS", lambda: test_voice_cloning(ad_script, speaker_sample, ad_audio_path))
    if ad_audio is None:
        return
    
    # Step 6: Test Audio Integration
    output_path = os.path.join(output_dir, 'test_integrated.wav')
    insertion_point = content_analysis['best_segment']['end'] if content_analysis.get('best_segment') else 30.0
    final_audio = test_step("Audio Integration", lambda: test_audio_integration(podcast_path, ad_audio, insertion_point, output_path))
    if final_audio is None:
        return
    
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED!")
    print(f"Output file: {final_audio}")
    print('='*60)

def test_ad_analyzer(script_path):
    """Test the ad analyzer"""
    from agents.ad_analyzer import AdScriptAnalyzer
    
    analyzer = AdScriptAnalyzer()
    result = analyzer.analyze(script_path)
    
    print(f"  Product: {result.get('product_name')}")
    print(f"  Problem: {result.get('problem_solved')}")
    print(f"  Keywords: {result.get('keywords', [])[:5]}")
    
    return result

def test_transcription(podcast_path):
    """Test podcast transcription"""
    from agents.podcast_analyzer import PodcastAnalyzer
    
    analyzer = PodcastAnalyzer()
    result = analyzer.transcribe(podcast_path)
    
    print(f"  Duration: {result.get('duration', 0):.2f}s")
    print(f"  Chunks: {len(result.get('chunks', []))}")
    print(f"  Text preview: {result.get('full_text', '')[:100]}...")
    
    return result

def test_content_analysis(transcription, ad_analysis):
    """Test content analysis for ad placement"""
    from agents.podcast_analyzer import PodcastAnalyzer
    
    analyzer = PodcastAnalyzer()
    result = analyzer.analyze_content(transcription, ad_analysis)
    
    best = result.get('best_segment', {})
    print(f"  Best segment: {best.get('start', 0):.2f}s - {best.get('end', 0):.2f}s")
    print(f"  Relevance: {best.get('relevance_score', 0):.3f}")
    
    return result

def test_speaker_sample(podcast_path):
    """Test speaker sample extraction"""
    from agents.podcast_analyzer import PodcastAnalyzer
    
    analyzer = PodcastAnalyzer()
    result = analyzer.get_speaker_sample(podcast_path)
    
    print(f"  Speaker sample path: {result}")
    print(f"  File exists: {os.path.exists(result)}")
    
    return result

def test_voice_cloning(ad_script, speaker_sample_path, output_path):
    """Test voice cloning / TTS"""
    from agents.voice_cloning import VoiceCloningAgent
    
    agent = VoiceCloningAgent()
    result = agent.generate_ad_audio(ad_script, speaker_sample_path, output_path)
    
    print(f"  Output path: {result}")
    print(f"  File exists: {os.path.exists(result)}")
    
    if os.path.exists(result):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(result)
        print(f"  Duration: {len(audio)/1000:.2f}s")
    
    return result

def test_audio_integration(podcast_path, ad_audio_path, insertion_point, output_path):
    """Test audio integration"""
    from agents.audio_integrator import AudioIntegrator
    
    integrator = AudioIntegrator()
    result = integrator.integrate_ad(
        podcast_path,
        ad_audio_path,
        insertion_point,
        output_path=output_path
    )
    
    print(f"  Output path: {result}")
    print(f"  File exists: {os.path.exists(result)}")
    
    if os.path.exists(result):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(result)
        print(f"  Duration: {len(audio)/1000:.2f}s")
    
    return result

if __name__ == '__main__':
    main()
