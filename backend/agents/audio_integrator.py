"""
Audio Integrator Agent
Integrates generated ad audio into the podcast with smooth transitions
"""
import os
import tempfile
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import numpy as np

class AudioIntegrator:
    def __init__(self):
        """Initialize the audio integrator"""
        self.default_transition_duration = 500  # ms
        self.default_fade_duration = 300  # ms
        
    def integrate_ad(
        self,
        podcast_path: str,
        ad_audio_path: str,
        insertion_timestamp: float,
        intro_audio_path: str = None,
        outro_audio_path: str = None,
        output_path: str = None,
        status_callback = None
    ) -> str:
        """
        Integrate ad audio into podcast at specified timestamp
        
        Args:
            podcast_path: Path to original podcast audio
            ad_audio_path: Path to generated ad audio
            insertion_timestamp: Timestamp in seconds where ad should be inserted
            intro_audio_path: Optional transition audio for ad intro
            outro_audio_path: Optional transition audio for ad outro
            output_path: Path for output file
            status_callback: Callback function for progress updates
        
        Returns:
            Path to integrated audio file
        """
        if status_callback:
            status_callback(80, "Integrating ad into podcast...")
        
        print(f"Loading podcast from: {podcast_path}")
        podcast = AudioSegment.from_file(podcast_path)
        
        print(f"Loading ad audio from: {ad_audio_path}")
        ad_audio = AudioSegment.from_file(ad_audio_path)
        
        # Normalize audio levels
        podcast = self._normalize_audio(podcast)
        ad_audio = self._normalize_audio(ad_audio)
        
        # Match ad audio format to podcast
        ad_audio = self._match_audio_format(ad_audio, podcast)
        
        # Calculate insertion point in milliseconds
        insertion_ms = int(insertion_timestamp * 1000)
        
        # Ensure insertion point is valid and leaves room for content after
        # Don't insert in the last 2 seconds to ensure smooth outro
        max_insertion = max(0, len(podcast) - 2000)
        insertion_ms = max(0, min(insertion_ms, max_insertion))
        
        print(f"Inserting ad at {insertion_ms/1000:.2f}s (podcast duration: {len(podcast)/1000:.2f}s)")
        
        # Split podcast at insertion point
        before_ad = podcast[:insertion_ms]
        after_ad = podcast[insertion_ms:]
        
        # Create transition segments
        intro_transition = self._create_transition(
            before_ad, 
            ad_audio, 
            intro_audio_path,
            transition_type='intro'
        )
        
        outro_transition = self._create_transition(
            ad_audio,
            after_ad,
            outro_audio_path,
            transition_type='outro'
        )
        
        # Helper function to safely append with crossfade
        def safe_append(base, segment, crossfade_ms):
            """Append audio segments with safe crossfade handling"""
            if len(segment) == 0:
                return base
            if len(base) == 0:
                return segment
            # Use minimum of desired crossfade and half of shorter segment
            safe_crossfade = min(crossfade_ms, len(base) // 2, len(segment) // 2)
            safe_crossfade = max(0, safe_crossfade)
            return base.append(segment, crossfade=safe_crossfade)
        
        # Assemble final audio
        # Format: [Before Ad] + [Intro Transition] + [Ad] + [Outro Transition] + [After Ad]
        
        final_audio = before_ad
        
        # Add intro transition
        if intro_transition and len(intro_transition) > 0:
            final_audio = safe_append(final_audio, intro_transition, self.default_fade_duration)
        
        # Add ad with crossfade
        final_audio = safe_append(final_audio, ad_audio, self.default_fade_duration)
        
        # Add outro transition
        if outro_transition and len(outro_transition) > 0:
            final_audio = safe_append(final_audio, outro_transition, self.default_fade_duration)
        
        # Add remaining podcast with crossfade
        final_audio = safe_append(final_audio, after_ad, self.default_fade_duration)
        
        # Normalize final output
        final_audio = self._normalize_audio(final_audio)
        
        # Determine output format from input
        input_format = os.path.splitext(podcast_path)[1].lower().replace('.', '')
        
        # Save output
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(
                suffix=f'.{input_format}', 
                delete=False
            ).name
        
        print(f"Exporting integrated podcast to: {output_path}")
        final_audio.export(output_path, format=input_format)
        
        return output_path
    
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels"""
        return normalize(audio)
    
    def _match_audio_format(self, source: AudioSegment, target: AudioSegment) -> AudioSegment:
        """Match source audio format to target"""
        # Match sample rate
        if source.frame_rate != target.frame_rate:
            source = source.set_frame_rate(target.frame_rate)
        
        # Match channels
        if source.channels != target.channels:
            if target.channels == 1:
                source = source.set_channels(1)
            else:
                source = source.set_channels(2)
        
        # Match sample width
        if source.sample_width != target.sample_width:
            source = source.set_sample_width(target.sample_width)
        
        return source
    
    def _create_transition(
        self,
        before_segment: AudioSegment,
        after_segment: AudioSegment,
        transition_audio_path: str = None,
        transition_type: str = 'intro'
    ) -> AudioSegment:
        """
        Create a smooth transition between segments
        """
        if transition_audio_path and os.path.exists(transition_audio_path):
            # Use provided transition audio
            transition = AudioSegment.from_file(transition_audio_path)
            transition = self._match_audio_format(transition, before_segment)
            return transition
        
        # Create a simple transition effect
        # Use a short segment of lower volume + fade
        
        if transition_type == 'intro':
            # Create a subtle "ducking" effect before ad
            transition_duration = self.default_transition_duration
            
            # Take last bit of before segment and reduce volume
            if len(before_segment) > transition_duration:
                duck_segment = before_segment[-transition_duration:]
                duck_segment = duck_segment - 3  # Reduce by 3dB
                duck_segment = duck_segment.fade_out(transition_duration)
                return duck_segment
        
        elif transition_type == 'outro':
            # Create fade in for resuming podcast
            transition_duration = self.default_transition_duration
            
            if len(after_segment) > transition_duration:
                rise_segment = after_segment[:transition_duration]
                rise_segment = rise_segment.fade_in(transition_duration)
                return rise_segment
        
        return None
    
    def integrate_ad_into_video(
        self,
        video_path: str,
        integrated_audio_path: str,
        output_path: str = None,
        status_callback = None
    ) -> str:
        """
        Replace audio in video with integrated audio
        """
        import subprocess
        
        if status_callback:
            status_callback(90, "Integrating audio into video...")
        
        if output_path is None:
            ext = os.path.splitext(video_path)[1]
            output_path = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
        
        # Use ffmpeg to replace audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', integrated_audio_path,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        
        print(f"Running ffmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def create_duck_effect(
        self,
        audio: AudioSegment,
        duck_start: int,
        duck_end: int,
        duck_amount: float = 6
    ) -> AudioSegment:
        """
        Create a ducking effect (lower volume) for a segment
        Useful for background music during ads
        """
        before = audio[:duck_start]
        during = audio[duck_start:duck_end]
        after = audio[duck_end:]
        
        # Apply ducking
        during = during - duck_amount  # Reduce by duck_amount dB
        
        # Apply fades
        fade_duration = min(500, len(during) // 4)
        during = during.fade_in(fade_duration).fade_out(fade_duration)
        
        return before + during + after


class TransitionGenerator:
    """Generate transition scripts for smooth ad integration"""
    
    INTRO_TEMPLATES = [
        "Speaking of {topic}, let me tell you about something that might interest you.",
        "You know, this reminds me of a product I've been using lately.",
        "Before we continue, I want to share something relevant with you.",
        "On the topic of {topic}, I have to mention our sponsor.",
        "That's a great segue into today's sponsor.",
    ]
    
    OUTRO_TEMPLATES = [
        "Alright, back to what we were discussing.",
        "Now, where were we? Right,",
        "Thanks for listening to that. Let's continue.",
        "So anyway, back to our conversation.",
        "Okay, let's get back to it.",
    ]
    
    @classmethod
    def generate_intro(cls, topic: str = None) -> str:
        """Generate ad intro transition text"""
        import random
        template = random.choice(cls.INTRO_TEMPLATES)
        if topic and '{topic}' in template:
            return template.format(topic=topic)
        return template.replace('{topic}', 'this')
    
    @classmethod
    def generate_outro(cls) -> str:
        """Generate ad outro transition text"""
        import random
        return random.choice(cls.OUTRO_TEMPLATES)


# Test function
if __name__ == "__main__":
    integrator = AudioIntegrator()
    print("Audio Integrator initialized successfully!")
    
    # Test transition generation
    intro = TransitionGenerator.generate_intro("financial planning")
    outro = TransitionGenerator.generate_outro()
    print(f"Sample intro: {intro}")
    print(f"Sample outro: {outro}")
