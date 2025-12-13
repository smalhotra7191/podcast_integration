"""
Orchestrator for Podcast Ad Integration Pipeline
Coordinates all agents to process podcast and integrate ads
"""
import os
import tempfile
from typing import Callable, Optional
from agents import (
    AdScriptAnalyzer,
    PodcastAnalyzer,
    VoiceCloningAgent,
    AudioIntegrator,
    TransitionGenerator
)

class PodcastAdOrchestrator:
    def __init__(
        self,
        podcast_path: str,
        ad_script_path: str,
        output_folder: str,
        job_id: str,
        status_callback: Optional[Callable[[int, str], None]] = None
    ):
        """
        Initialize the orchestrator
        
        Args:
            podcast_path: Path to the podcast audio/video file
            ad_script_path: Path to the ad script text file
            output_folder: Folder to save output files
            job_id: Unique identifier for this job
            status_callback: Callback function(progress, message) for status updates
        """
        self.podcast_path = podcast_path
        self.ad_script_path = ad_script_path
        self.output_folder = output_folder
        self.job_id = job_id
        self.status_callback = status_callback or (lambda p, m: None)
        
        # Initialize agents
        self.ad_analyzer = AdScriptAnalyzer()
        self.podcast_analyzer = PodcastAnalyzer()
        self.voice_cloner = VoiceCloningAgent()
        self.audio_integrator = AudioIntegrator()
        
        # Determine if input is video
        self.is_video = self._is_video_file(podcast_path)
        
        # Create job output folder
        self.job_output_folder = os.path.join(output_folder, job_id)
        os.makedirs(self.job_output_folder, exist_ok=True)
    
    def _is_video_file(self, path: str) -> bool:
        """Check if file is a video"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        return os.path.splitext(path)[1].lower() in video_extensions
    
    def _update_status(self, progress: int, message: str):
        """Update job status"""
        print(f"[{progress}%] {message}")
        self.status_callback(progress, message)
    
    def process(self) -> str:
        """
        Run the full pipeline to integrate ad into podcast
        
        Returns:
            Path to the output file
        """
        self._update_status(5, "Starting pipeline...")
        
        # Step 1: Analyze ad script
        self._update_status(10, "Analyzing ad script...")
        ad_analysis = self.ad_analyzer.analyze(self.ad_script_path)
        
        print(f"Ad Analysis Results:")
        print(f"  Product: {ad_analysis['product_name']}")
        print(f"  Problem: {ad_analysis['problem_solved']}")
        print(f"  Keywords: {ad_analysis['keywords'][:5]}")
        
        # Step 2: Extract audio from video if needed
        audio_path = self.podcast_path
        if self.is_video:
            self._update_status(15, "Extracting audio from video...")
            audio_path = self.podcast_analyzer.extract_audio(self.podcast_path)
        
        # Step 3: Transcribe and analyze podcast
        self._update_status(20, "Transcribing podcast...")
        transcription = self.podcast_analyzer.transcribe(
            audio_path,
            status_callback=self.status_callback
        )
        
        self._update_status(40, "Analyzing podcast content for ad placement...")
        content_analysis = self.podcast_analyzer.analyze_content(
            transcription,
            ad_analysis,
            status_callback=self.status_callback
        )
        
        # Step 4: Determine best insertion point
        insertion_point = self._select_insertion_point(content_analysis)
        print(f"Selected insertion point: {insertion_point['timestamp']:.2f}s")
        print(f"Relevance score: {insertion_point['relevance_score']:.3f}")
        
        # Step 5: Extract speaker sample for voice cloning
        self._update_status(50, "Extracting speaker voice sample...")
        speaker_sample = self.podcast_analyzer.get_speaker_sample(audio_path)
        
        # Step 6: Generate ad script with transitions
        self._update_status(55, "Preparing ad script with transitions...")
        full_ad_script = self._create_ad_with_transitions(
            ad_analysis,
            content_analysis,
            insertion_point
        )
        
        # Step 7: Generate ad audio in speaker's voice
        self._update_status(60, "Generating ad audio in podcaster's voice...")
        ad_audio_path = os.path.join(self.job_output_folder, 'generated_ad.wav')
        ad_audio_path = self.voice_cloner.generate_ad_audio(
            full_ad_script,
            speaker_sample,
            ad_audio_path,
            status_callback=self.status_callback
        )
        
        # Step 8: Integrate ad into podcast
        self._update_status(80, "Integrating ad into podcast...")
        
        # Determine output path
        ext = os.path.splitext(self.podcast_path)[1]
        if self.is_video:
            # For video, we'll first create integrated audio, then merge with video
            integrated_audio_path = os.path.join(
                self.job_output_folder, 
                'integrated_audio.wav'
            )
            
            integrated_audio_path = self.audio_integrator.integrate_ad(
                audio_path,
                ad_audio_path,
                insertion_point['timestamp'],
                output_path=integrated_audio_path
            )
            
            self._update_status(90, "Merging audio with video...")
            output_path = os.path.join(
                self.job_output_folder,
                f'integrated_podcast{ext}'
            )
            
            output_path = self.audio_integrator.integrate_ad_into_video(
                self.podcast_path,
                integrated_audio_path,
                output_path,
                status_callback=self.status_callback
            )
        else:
            # For audio-only, directly integrate
            output_path = os.path.join(
                self.job_output_folder,
                f'integrated_podcast{ext}'
            )
            
            output_path = self.audio_integrator.integrate_ad(
                self.podcast_path,
                ad_audio_path,
                insertion_point['timestamp'],
                output_path=output_path
            )
        
        # Clean up temp files
        if self.is_video and audio_path != self.podcast_path:
            try:
                os.unlink(audio_path)
            except:
                pass
        
        try:
            os.unlink(speaker_sample)
        except:
            pass
        
        self._update_status(100, "Processing complete!")
        
        return output_path
    
    def _select_insertion_point(self, content_analysis: dict) -> dict:
        """Select the best insertion point for the ad"""
        candidates = content_analysis.get('insertion_candidates', [])
        
        if not candidates:
            # Fallback: Insert at 1/3 of the podcast duration
            duration = content_analysis.get('duration', 300)
            return {
                'timestamp': duration / 3,
                'relevance_score': 0.5,
                'is_natural_break': False,
                'context': ''
            }
        
        # Prefer natural breaks with high relevance
        for candidate in candidates:
            if candidate.get('is_natural_break', False):
                return candidate
        
        # Otherwise, use highest relevance
        return candidates[0]
    
    def _create_ad_with_transitions(
        self,
        ad_analysis: dict,
        content_analysis: dict,
        insertion_point: dict
    ) -> str:
        """Create the full ad script with intro and outro transitions"""
        
        # Generate contextual intro
        topic = ad_analysis.get('problem_solved', '')
        intro = TransitionGenerator.generate_intro(topic)
        
        # Get the ad script
        ad_script = ad_analysis.get('full_script', '')
        
        # Generate outro
        outro = TransitionGenerator.generate_outro()
        
        # Combine
        full_script = f"{intro}\n\n{ad_script}\n\n{outro}"
        
        return full_script


# Test function
if __name__ == "__main__":
    print("Orchestrator module loaded successfully!")
    print("To use, create a PodcastAdOrchestrator instance and call process()")
