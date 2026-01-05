"""Quick test with full podcast"""
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.podcast_analyzer import PodcastAnalyzer
from agents.ad_analyzer import AdScriptAnalyzer

# Initialize
print("Loading models...")
p = PodcastAnalyzer(speed_mode='fast')
a = AdScriptAnalyzer()

# Analyze ad
print("\nAnalyzing ad...")
ad_script_path = '../samples/sample_ad_script.txt'
ad = a.analyze(ad_script_path)
print(f"Problem: {ad['problem_solved']}")
print(f"Keywords: {ad.get('problem_keywords', [])[:5]}")

# Analyze full podcast
podcast_path = '../samples/Ep1_full_without_music.wav'
print(f"\nTranscribing full podcast...")
transcription = p.transcribe(podcast_path)
print(f"Transcription done: {len(transcription['chunks'])} chunks, {transcription['duration']:.1f}s")

print("\nAnalyzing content for ad placement...")
result = p.analyze_content(transcription, ad, audio_path=podcast_path)

print(f"\n{'='*60}")
print(f"Placement method: {result['placement_method']}")
print(f"Number of candidates: {len(result['insertion_candidates'])}")
print(f"{'='*60}")

print("\nTop 3 candidates:")
for i, c in enumerate(result['insertion_candidates'][:3], 1):
    print(f"\n{i}. Timestamp: {c['timestamp']:.1f}s")
    print(f"   Score: {c['relevance_score']:.3f}")
    print(f"   Natural break: {c['is_natural_break']}")
    print(f"   Reason: {c['placement_reason']}")
    print(f"   Context: ...{c['boundary_context'][-80:]}...")
