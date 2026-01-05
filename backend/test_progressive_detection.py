"""Test progressive speaker detection"""
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import PodcastAnalyzer

print('Testing progressive speaker detection...')
analyzer = PodcastAnalyzer(speed_mode='fast')

found_speakers = []

def on_speaker_found(speaker):
    print(f'  -> Found speaker {speaker["id"]}: {speaker["label"]} ({speaker["total_speaking_time"]:.1f}s)')
    found_speakers.append(speaker)
    return True  # Continue detection

os.makedirs('./test_outputs/test_progressive', exist_ok=True)

result = analyzer.detect_speakers_progressive(
    '../samples/Ep1_full_without_music.wav',
    './test_outputs/test_progressive',
    on_speaker_found=on_speaker_found,
    status_callback=lambda p, m: print(f'[{p}%] {m}')
)

print(f'\nFinal result: {result["num_speakers"]} speaker(s) found')
for s in result['speakers']:
    print(f'  - {s["label"]}: {s["total_speaking_time"]:.1f}s')
