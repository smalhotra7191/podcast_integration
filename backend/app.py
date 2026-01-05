"""
Flask Backend API for Podcast Ad Integration
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import threading
import traceback
from werkzeug.utils import secure_filename
from orchestrator import PodcastAdOrchestrator

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
CHUNKS_FOLDER = os.path.join(os.path.dirname(__file__), 'chunks')
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_TEXT_EXTENSIONS = {'txt', 'md'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHUNKS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max file size (for 4hr videos)

# Store job status
jobs = {}

def allowed_file(filename, file_type):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'media':
        return ext in ALLOWED_AUDIO_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)
    elif file_type == 'text':
        return ext in ALLOWED_TEXT_EXTENSIONS
    return False

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return 'video'
    return None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Podcast Ad Integration API is running'})

@app.route('/api/debug/jobs', methods=['GET'])
def debug_list_jobs():
    """Debug endpoint to list all jobs"""
    result = {}
    for job_id, job_data in jobs.items():
        result[job_id] = {
            'status': job_data.get('status'),
            'message': job_data.get('message'),
            'podcast_path': job_data.get('podcast_path'),
            'has_speaker_info': 'speaker_info' in job_data
        }
    return jsonify({'jobs_count': len(jobs), 'jobs': result})

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload podcast media and ad script"""
    try:
        if 'podcast' not in request.files:
            return jsonify({'error': 'No podcast file provided'}), 400
        
        if 'adScript' not in request.files:
            return jsonify({'error': 'No ad script file provided'}), 400
        
        podcast_file = request.files['podcast']
        ad_script_file = request.files['adScript']
        workflow = request.form.get('workflow', 'full')  # 'full' or 'ad_only'
        
        if podcast_file.filename == '':
            return jsonify({'error': 'No podcast file selected'}), 400
        
        if ad_script_file.filename == '':
            return jsonify({'error': 'No ad script file selected'}), 400
        
        if not allowed_file(podcast_file.filename, 'media'):
            return jsonify({'error': 'Invalid podcast file format'}), 400
        
        if not allowed_file(ad_script_file.filename, 'text'):
            return jsonify({'error': 'Invalid ad script file format'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Save files
        podcast_filename = secure_filename(podcast_file.filename)
        ad_script_filename = secure_filename(ad_script_file.filename)
        
        podcast_path = os.path.join(job_folder, podcast_filename)
        ad_script_path = os.path.join(job_folder, ad_script_filename)
        
        podcast_file.save(podcast_path)
        ad_script_file.save(ad_script_path)
        
        media_type = get_file_type(podcast_filename)
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'uploaded',
            'podcast_path': podcast_path,
            'ad_script_path': ad_script_path,
            'media_type': media_type,
            'progress': 0,
            'message': 'Files uploaded successfully',
            'workflow': workflow
        }
        
        return jsonify({
            'job_id': job_id,
            'message': 'Files uploaded successfully',
            'media_type': media_type,
            'workflow': workflow
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/<job_id>', methods=['POST'])
def process_podcast(job_id):
    """Start processing the podcast with ad integration"""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        # Allow processing from various valid states
        valid_states = ['uploaded', 'speakers_detected', 'speaker_selected', 'error']
        if job['status'] not in valid_states:
            return jsonify({'error': 'Job already processing or completed'}), 400
        
        # Get speed mode and workflow from request (default to 'fast' for better UX)
        # Handle both JSON and non-JSON requests
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}
        speed_mode = data.get('speed_mode', 'fast')  # 'fast', 'balanced', 'accurate'
        workflow = data.get('workflow', job.get('workflow', 'full'))  # 'full' or 'ad_only'
        
        # Update job status
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 5
        jobs[job_id]['message'] = 'Starting processing...'
        jobs[job_id]['speed_mode'] = speed_mode
        jobs[job_id]['workflow'] = workflow
        
        # Initialize cancellation flag
        job_cancellation[job_id] = False
        
        # Get selected speaker if available
        selected_speaker = job.get('selected_speaker', None)
        audio_path = job.get('audio_path', None)  # Pre-extracted during speaker detection
        
        # Get pre-saved speaker sample path if speaker was selected
        speaker_sample_path = None
        speaker_info = job.get('speaker_info', {})
        if selected_speaker is not None and 'speakers' in speaker_info:
            for speaker in speaker_info['speakers']:
                if speaker['id'] == selected_speaker:
                    speaker_sample_path = speaker.get('sample_path')
                    print(f"Using pre-saved speaker sample for speaker {selected_speaker}: {speaker_sample_path}")
                    break
        
        # Start processing in background thread
        def process_in_background():
            try:
                # Check if cancelled before starting
                if job_cancellation.get(job_id, False):
                    jobs[job_id]['status'] = 'cancelled'
                    jobs[job_id]['message'] = 'Processing was cancelled'
                    return
                
                orchestrator = PodcastAdOrchestrator(
                    podcast_path=job['podcast_path'],
                    ad_script_path=job['ad_script_path'],
                    output_folder=OUTPUT_FOLDER,
                    job_id=job_id,
                    status_callback=lambda progress, message: update_job_status(job_id, progress, message),
                    speed_mode=speed_mode,
                    selected_speaker=selected_speaker,
                    audio_path=audio_path,
                    workflow=workflow,
                    speaker_sample_path=speaker_sample_path,
                    cancellation_check=lambda: job_cancellation.get(job_id, False)
                )
                
                output_path = orchestrator.process()
                
                # Check if cancelled after processing
                if job_cancellation.get(job_id, False):
                    jobs[job_id]['status'] = 'cancelled'
                    jobs[job_id]['message'] = 'Processing was cancelled'
                    return
                
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['progress'] = 100
                jobs[job_id]['message'] = 'Processing completed!'
                jobs[job_id]['output_path'] = output_path
                
                # Store voice similarity score
                if orchestrator.voice_similarity_score:
                    jobs[job_id]['voice_similarity'] = orchestrator.voice_similarity_score
                
                # Store ad placement info (for full workflow)
                if orchestrator.ad_placement_info:
                    jobs[job_id]['ad_placement'] = orchestrator.ad_placement_info
                
            except Exception as e:
                print(f"Processing error: {str(e)}")
                traceback.print_exc()
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['message'] = str(e)
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': f'Processing started (workflow: {workflow}, speed mode: {speed_mode})'
        })
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
        return jsonify({'error': str(e)}), 500

def update_job_status(job_id, progress, message):
    """Callback to update job status during processing"""
    if job_id in jobs:
        jobs[job_id]['progress'] = progress
        jobs[job_id]['message'] = message

@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job"""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        current_status = job['status']
        
        # Set cancellation flags
        if job_id in speaker_detection_cancellation:
            speaker_detection_cancellation[job_id] = True
        
        job_cancellation[job_id] = True
        
        # Update job status
        jobs[job_id]['status'] = 'cancelled'
        jobs[job_id]['message'] = 'Processing cancelled by user'
        
        print(f"Job {job_id}: Cancelled (was in status: {current_status})")
        
        return jsonify({
            'job_id': job_id,
            'status': 'cancelled',
            'message': 'Job cancelled successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a processing job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'media_type': job.get('media_type'),
        'workflow': job.get('workflow', 'full')
    }
    
    # Include voice similarity score if available
    if 'voice_similarity' in job:
        response['voice_similarity'] = job['voice_similarity']
    
    # Include ad placement info if available
    if 'ad_placement' in job:
        response['ad_placement'] = job['ad_placement']
    
    return jsonify(response)

@app.route('/api/download/<job_id>', methods=['GET'])
def download_result(job_id):
    """Download the processed podcast or generated ad"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    if 'output_path' not in job:
        return jsonify({'error': 'Output file not found'}), 404
    
    output_path = job['output_path']
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404
    
    # Determine filename based on workflow
    workflow = job.get('workflow', 'full')
    if workflow == 'ad_only':
        filename = f'generated_ad_{job_id}{os.path.splitext(output_path)[1]}'
    else:
        filename = f'integrated_podcast_{job_id}{os.path.splitext(output_path)[1]}'
    
    return send_file(
        output_path,
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/analysis/<job_id>', methods=['GET'])
def get_analysis(job_id):
    """Get the analysis results for a job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    return jsonify({
        'job_id': job_id,
        'ad_analysis': job.get('ad_analysis', {}),
        'podcast_analysis': job.get('podcast_analysis', {}),
        'integration_point': job.get('integration_point', {})
    })

# ============== Cancellation Infrastructure ==============

# Store for cancellation tokens
speaker_detection_cancellation = {}
job_cancellation = {}  # For cancelling processing jobs

@app.route('/api/detect-speakers/<job_id>', methods=['POST'])
def detect_speakers(job_id):
    """Start speaker detection in background and return immediately"""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        if job['status'] != 'uploaded':
            return jsonify({'error': 'Job must be in uploaded state'}), 400
        
        # Update status
        jobs[job_id]['status'] = 'detecting_speakers'
        jobs[job_id]['message'] = 'Starting speaker detection...'
        jobs[job_id]['speakers_found'] = []  # List of speakers found so far
        jobs[job_id]['speaker_detection_complete'] = False
        
        # Initialize cancellation token
        speaker_detection_cancellation[job_id] = False
        
        # Create output folder for speaker samples
        job_folder = os.path.dirname(job['podcast_path'])
        
        def detect_in_background():
            try:
                from agents import PodcastAnalyzer
                analyzer = PodcastAnalyzer(speed_mode='fast')
                
                # Extract audio if video
                audio_path = job['podcast_path']
                if job.get('media_type') == 'video':
                    audio_path = analyzer.extract_audio(job['podcast_path'])
                    jobs[job_id]['audio_path'] = audio_path
                else:
                    jobs[job_id]['audio_path'] = audio_path
                
                # Use progressive speaker detection
                def on_speaker_found(speaker_info):
                    """Callback when a speaker is found"""
                    # Check if cancelled
                    if speaker_detection_cancellation.get(job_id, False):
                        return False  # Signal to stop
                    
                    jobs[job_id]['speakers_found'].append(speaker_info)
                    jobs[job_id]['message'] = f'Found {len(jobs[job_id]["speakers_found"])} speaker(s)...'
                    print(f"Job {job_id}: Found speaker {speaker_info['id']}")
                    return True  # Continue detection
                
                # Run progressive detection
                final_result = analyzer.detect_speakers_progressive(
                    audio_path,
                    job_folder,
                    on_speaker_found=on_speaker_found,
                    status_callback=lambda p, m: update_job_status(job_id, p, m)
                )
                
                # Check if cancelled
                if speaker_detection_cancellation.get(job_id, False):
                    print(f"Job {job_id}: Speaker detection was cancelled")
                    return
                
                # Store final speaker info
                jobs[job_id]['speaker_info'] = final_result
                jobs[job_id]['speaker_detection_complete'] = True
                jobs[job_id]['status'] = 'speakers_detected'
                jobs[job_id]['message'] = f'Detected {final_result["num_speakers"]} speaker(s)'
                
                # IMPORTANT: Update speakers_found with the FINAL speaker list
                # The preliminary speakers_found may have different/fewer speakers
                # than the final clustering result
                jobs[job_id]['speakers_found'] = final_result.get('speakers', [])
                
            except Exception as e:
                traceback.print_exc()
                if not speaker_detection_cancellation.get(job_id, False):
                    jobs[job_id]['status'] = 'error'
                    jobs[job_id]['message'] = str(e)
        
        # Start background thread
        thread = threading.Thread(target=detect_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'detecting_speakers',
            'message': 'Speaker detection started'
        })
        
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/speaker-status/<job_id>', methods=['GET'])
def get_speaker_status(job_id):
    """Get current speaker detection status and any speakers found so far"""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        speakers_found = job.get('speakers_found', [])
        detection_complete = job.get('speaker_detection_complete', False)
        
        return jsonify({
            'job_id': job_id,
            'status': job['status'],
            'message': job.get('message', ''),
            'progress': job.get('progress', 0),
            'speakers_found': speakers_found,
            'num_speakers': len(speakers_found),
            'detection_complete': detection_complete,
            'requires_selection': len(speakers_found) > 1
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speaker-sample/<job_id>/<int:speaker_id>', methods=['GET'])
def get_speaker_sample(job_id, speaker_id):
    """Get audio sample for a specific speaker"""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        # First check final speaker_info (for completed detection)
        speaker_info = job.get('speaker_info', {})
        speakers = speaker_info.get('speakers', [])
        
        # Also check speakers_found (for progressive/in-progress detection)
        speakers_found = job.get('speakers_found', [])
        
        # Find the speaker in either list
        speaker = None
        for s in speakers:
            if s['id'] == speaker_id:
                speaker = s
                break
        
        # If not found in final speakers, check in-progress speakers
        if not speaker:
            for s in speakers_found:
                if s['id'] == speaker_id:
                    speaker = s
                    break
        
        if not speaker:
            return jsonify({'error': 'Speaker not found'}), 404
        
        sample_path = speaker.get('sample_path')
        if not sample_path or not os.path.exists(sample_path):
            return jsonify({'error': 'Speaker sample not found'}), 404
        
        return send_file(
            sample_path,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=f'speaker_{speaker_id}_sample.wav'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/select-speaker/<job_id>', methods=['POST'])
def select_speaker(job_id):
    """Select which speaker's voice to use for the ad"""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        # Get selected speaker from request
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}
        
        speaker_id = data.get('speaker_id', 0)
        
        # Cancel ongoing speaker detection if still running
        if job_id in speaker_detection_cancellation:
            speaker_detection_cancellation[job_id] = True
            print(f"Job {job_id}: Cancelling speaker detection - user selected speaker {speaker_id}")
        
        # Get the selected speaker's sample path
        speakers_found = job.get('speakers_found', [])
        selected_speaker_info = None
        for speaker in speakers_found:
            if speaker['id'] == speaker_id:
                selected_speaker_info = speaker
                break
        
        # Store selection
        jobs[job_id]['selected_speaker'] = speaker_id
        jobs[job_id]['selected_speaker_info'] = selected_speaker_info
        jobs[job_id]['status'] = 'speaker_selected'
        jobs[job_id]['message'] = f'Speaker {speaker_id + 1} selected for ad voice'
        
        # Store speaker info if not already done (from partial detection)
        if 'speaker_info' not in jobs[job_id] and speakers_found:
            jobs[job_id]['speaker_info'] = {
                'num_speakers': len(speakers_found),
                'speakers': speakers_found,
                'requires_selection': len(speakers_found) > 1
            }
        
        return jsonify({
            'job_id': job_id,
            'selected_speaker': speaker_id,
            'message': f'Speaker {speaker_id + 1} selected'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============== Chunked Upload Endpoints ==============

@app.route('/api/upload/init', methods=['POST'])
def init_chunked_upload():
    """Initialize a chunked upload session"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        file_size = data.get('fileSize')
        file_type = data.get('fileType')  # 'podcast' or 'adScript'
        
        if not filename or not file_size:
            return jsonify({'error': 'Missing filename or fileSize'}), 400
        
        # Generate upload ID
        upload_id = str(uuid.uuid4())
        
        # Create chunks folder for this upload
        upload_chunks_folder = os.path.join(CHUNKS_FOLDER, upload_id)
        os.makedirs(upload_chunks_folder, exist_ok=True)
        
        # Store upload metadata
        if 'uploads' not in app.config:
            app.config['uploads'] = {}
        
        app.config['uploads'][upload_id] = {
            'filename': secure_filename(filename),
            'file_size': file_size,
            'file_type': file_type,
            'chunks_received': 0,
            'total_chunks': 0,
            'chunks_folder': upload_chunks_folder
        }
        
        return jsonify({
            'upload_id': upload_id,
            'message': 'Upload initialized'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/chunk', methods=['POST'])
def upload_chunk():
    """Upload a single chunk"""
    try:
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex'))
        total_chunks = int(request.form.get('totalChunks'))
        
        if 'chunk' not in request.files:
            return jsonify({'error': 'No chunk provided'}), 400
        
        chunk = request.files['chunk']
        
        if upload_id not in app.config.get('uploads', {}):
            print(f"Invalid upload ID in chunk upload: {upload_id}")
            return jsonify({'error': 'Invalid upload ID'}), 400
        
        upload_info = app.config['uploads'][upload_id]
        upload_info['total_chunks'] = total_chunks
        
        # Save chunk
        chunk_path = os.path.join(upload_info['chunks_folder'], f'chunk_{chunk_index:06d}')
        chunk.save(chunk_path)
        
        upload_info['chunks_received'] += 1
        
        # Log progress for final chunks
        if chunk_index >= total_chunks - 3:
            print(f"Chunk {chunk_index + 1}/{total_chunks} uploaded for {upload_info['filename']}")
        
        return jsonify({
            'message': f'Chunk {chunk_index + 1}/{total_chunks} uploaded',
            'chunks_received': upload_info['chunks_received']
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/complete', methods=['POST', 'OPTIONS'])
def complete_chunked_upload():
    """Complete chunked upload by merging chunks"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        job_id = data.get('jobId')
        
        print(f"Completing chunked upload: upload_id={upload_id}, job_id={job_id}")
        
        if upload_id not in app.config.get('uploads', {}):
            print(f"Invalid upload ID: {upload_id}")
            print(f"Available uploads: {list(app.config.get('uploads', {}).keys())}")
            return jsonify({'error': 'Invalid upload ID'}), 400
        
        upload_info = app.config['uploads'][upload_id]
        print(f"Upload info: {upload_info}")
        
        # Create job folder if needed
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Merge chunks into final file
        final_path = os.path.join(job_folder, upload_info['filename'])
        print(f"Merging {upload_info['total_chunks']} chunks to: {final_path}")
        
        with open(final_path, 'wb') as final_file:
            for i in range(upload_info['total_chunks']):
                chunk_path = os.path.join(upload_info['chunks_folder'], f'chunk_{i:06d}')
                with open(chunk_path, 'rb') as chunk_file:
                    final_file.write(chunk_file.read())
                # Delete chunk after merging
                os.remove(chunk_path)
        
        print(f"Merge complete: {final_path}")
        
        # Clean up chunks folder
        try:
            os.rmdir(upload_info['chunks_folder'])
        except:
            pass
        
        # Clean up upload info
        del app.config['uploads'][upload_id]
        
        return jsonify({
            'message': 'Upload completed',
            'file_path': final_path,
            'file_type': upload_info['file_type']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/chunked', methods=['POST'])
def upload_files_chunked():
    """Finalize job after chunked uploads are complete"""
    try:
        data = request.get_json()
        job_id = data.get('jobId')
        podcast_path = data.get('podcastPath')
        ad_script_path = data.get('adScriptPath')
        workflow = data.get('workflow', 'full')  # 'full' or 'ad_only'
        
        if not job_id or not podcast_path or not ad_script_path:
            return jsonify({'error': 'Missing required fields'}), 400
        
        media_type = get_file_type(os.path.basename(podcast_path))
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'uploaded',
            'podcast_path': podcast_path,
            'ad_script_path': ad_script_path,
            'media_type': media_type,
            'progress': 0,
            'message': 'Files uploaded successfully',
            'workflow': workflow
        }
        
        return jsonify({
            'job_id': job_id,
            'message': 'Files uploaded successfully',
            'media_type': media_type,
            'workflow': workflow
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
