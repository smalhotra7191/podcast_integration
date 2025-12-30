"""Test speaker detection"""
import sys
sys.path.insert(0, '.')
from agents.podcast_analyzer import PodcastAnalyzer
import os

# Test with the sample podcast
audio_path = r'c:\Users\sameer\OneDrive\4_Projects\Naarad\Product\samples\Ep1_full_without_music.wav'
output_folder = r'c:\Users\sameer\OneDrive\4_Projects\Naarad\Product\backend\test_outputs'
os.makedirs(output_folder, exist_ok=True)

print('Creating analyzer...')
analyzer = PodcastAnalyzer(speed_mode='fast')

print('Detecting speakers...')
result = analyzer.detect_speakers(audio_path, output_folder)

print('Result:')
print(f'  num_speakers: {result["num_speakers"]}')
print(f'  requires_selection: {result["requires_selection"]}')
for s in result['speakers']:
    print(f'  Speaker {s["id"]}: sample_path={s.get("sample_path")}, duration={s.get("total_speaking_time")}s')
