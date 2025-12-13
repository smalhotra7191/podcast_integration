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
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_TEXT_EXTENSIONS = {'txt', 'md'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

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
            'message': 'Files uploaded successfully'
        }
        
        return jsonify({
            'job_id': job_id,
            'message': 'Files uploaded successfully',
            'media_type': media_type
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
        
        if job['status'] not in ['uploaded', 'error']:
            return jsonify({'error': 'Job already processing or completed'}), 400
        
        # Update job status
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 5
        jobs[job_id]['message'] = 'Starting processing...'
        
        # Start processing in background thread
        def process_in_background():
            try:
                orchestrator = PodcastAdOrchestrator(
                    podcast_path=job['podcast_path'],
                    ad_script_path=job['ad_script_path'],
                    output_folder=OUTPUT_FOLDER,
                    job_id=job_id,
                    status_callback=lambda progress, message: update_job_status(job_id, progress, message)
                )
                
                output_path = orchestrator.process()
                
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['progress'] = 100
                jobs[job_id]['message'] = 'Processing completed!'
                jobs[job_id]['output_path'] = output_path
                
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
            'message': 'Processing started'
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

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a processing job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'media_type': job.get('media_type')
    })

@app.route('/api/download/<job_id>', methods=['GET'])
def download_result(job_id):
    """Download the processed podcast"""
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
    
    return send_file(
        output_path,
        as_attachment=True,
        download_name=f'integrated_podcast_{job_id}{os.path.splitext(output_path)[1]}'
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
