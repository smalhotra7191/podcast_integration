"""Test speaker similarity between detected speakers"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import PodcastAnalyzer

print('Testing speaker similarity...')
analyzer = PodcastAnalyzer(speed_mode='fast')

# Get speaker samples from the last test run
test_output = './test_outputs/test_progressive'
speaker_0_sample = os.path.join(test_output, 'speaker_0_sample.wav')
speaker_1_sample = os.path.join(test_output, 'speaker_1_sample.wav')

if os.path.exists(speaker_0_sample) and os.path.exists(speaker_1_sample):
    print(f'\nComparing speaker samples:')
    print(f'  Speaker 0: {speaker_0_sample}')
    print(f'  Speaker 1: {speaker_1_sample}')
    
    # Get embeddings for both speakers
    emb0 = analyzer._get_speaker_embedding(speaker_0_sample)
    emb1 = analyzer._get_speaker_embedding(speaker_1_sample)
    
    # Normalize
    emb0 = emb0 / np.linalg.norm(emb0)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    # Compute cosine similarity
    similarity = np.dot(emb0, emb1)
    cosine_distance = 1 - similarity
    
    print(f'\nCosine Similarity: {similarity:.4f}')
    print(f'Cosine Distance: {cosine_distance:.4f}')
    
    if similarity > 0.85:
        print(f'\n=> These are likely the SAME speaker (similarity > 0.85)')
    elif similarity > 0.7:
        print(f'\n=> These MIGHT be the same speaker (similarity 0.7-0.85)')
    else:
        print(f'\n=> These are likely DIFFERENT speakers (similarity < 0.7)')
else:
    print('Speaker samples not found. Run test_progressive_detection.py first.')
