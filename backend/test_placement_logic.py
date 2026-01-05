"""
Test script for the new stepwise ad placement logic.
Tests:
1. Ad analysis - extracting problem being solved
2. Problem mention search in podcast
3. Insertion point selection
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.ad_analyzer import AdScriptAnalyzer
from agents.podcast_analyzer import PodcastAnalyzer

def test_ad_placement_logic():
    print("=" * 70)
    print("TESTING NEW STEPWISE AD PLACEMENT LOGIC")
    print("=" * 70)
    
    # Paths
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    ad_script_path = os.path.join(samples_dir, 'sample_ad_script.txt')
    # Use the shorter clip for faster testing
    podcast_path = os.path.join(samples_dir, 'Ep1_AClip1.wav')
    
    print(f"\nAd Script: {ad_script_path}")
    print(f"Podcast: {podcast_path}")
    
    # ========== STEP 1: Ad Analysis ==========
    print("\n" + "=" * 70)
    print("STEP 1: AD ANALYSIS - Extracting Problem Being Solved")
    print("=" * 70)
    
    ad_analyzer = AdScriptAnalyzer()
    ad_analysis = ad_analyzer.analyze(ad_script_path)
    
    print(f"\n✓ Product Name: {ad_analysis['product_name']}")
    print(f"✓ Problem Solved: {ad_analysis['problem_solved']}")
    print(f"✓ Problem Keywords: {ad_analysis.get('problem_keywords', [])}")
    print(f"✓ General Keywords: {ad_analysis['keywords'][:5]}")
    print(f"✓ Target Audience: {ad_analysis['target_audience']}")
    
    # ========== STEP 2: Podcast Transcription ==========
    print("\n" + "=" * 70)
    print("STEP 2: PODCAST TRANSCRIPTION")
    print("=" * 70)
    
    podcast_analyzer = PodcastAnalyzer(speed_mode='fast')
    
    def status_callback(progress, message):
        print(f"  [{progress:3d}%] {message}")
    
    print("\nTranscribing podcast...")
    transcription = podcast_analyzer.transcribe(podcast_path, status_callback)
    
    print(f"\n✓ Duration: {transcription['duration']:.1f}s")
    print(f"✓ Segments: {len(transcription['chunks'])}")
    print(f"✓ Sample text: {transcription['full_text'][:200]}...")
    
    # ========== STEP 3: Content Analysis with New Logic ==========
    print("\n" + "=" * 70)
    print("STEP 3: CONTENT ANALYSIS - Problem Search & Similarity")
    print("=" * 70)
    
    print("\nAnalyzing content for ad placement...")
    content_analysis = podcast_analyzer.analyze_content(
        transcription,
        ad_analysis,
        status_callback,
        podcast_path
    )
    
    print(f"\n✓ Placement Method: {content_analysis.get('placement_method', 'unknown')}")
    print(f"✓ Number of insertion candidates: {len(content_analysis.get('insertion_candidates', []))}")
    
    # Show insertion candidates
    candidates = content_analysis.get('insertion_candidates', [])
    if candidates:
        print("\nTop Insertion Candidates:")
        for i, candidate in enumerate(candidates[:3]):
            print(f"\n  Candidate {i+1}:")
            print(f"    - Timestamp: {candidate['timestamp']:.2f}s")
            print(f"    - Relevance Score: {candidate['relevance_score']:.3f}")
            print(f"    - Natural Break: {candidate.get('is_natural_break', False)}")
            print(f"    - Reason: {candidate.get('placement_reason', 'N/A')}")
            if candidate.get('boundary_context'):
                print(f"    - Context: ...{candidate['boundary_context'][-80:]}")
    else:
        print("\n⚠ No insertion candidates found!")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if candidates:
        best = candidates[0]
        print(f"\n✓ Best insertion point: {best['timestamp']:.2f}s")
        print(f"✓ Method used: {content_analysis.get('placement_method', 'unknown')}")
        if content_analysis.get('placement_method') == 'problem_match':
            print("  → Ad will be placed after problem discussion in the podcast")
        else:
            print("  → Problem not found, using semantic similarity fallback")
        print(f"✓ Placement reason: {best.get('placement_reason', 'N/A')}")
    else:
        print("\n✗ Could not find suitable insertion point")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return content_analysis

if __name__ == "__main__":
    test_ad_placement_logic()
