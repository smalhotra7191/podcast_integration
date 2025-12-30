// AUTO-REBUILD: 2025-12-14 00:43:51
import React, { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import {
  Upload,
  FileAudio,
  FileText,
  Loader2,
  Download,
  CheckCircle2,
  XCircle,
  Mic,
  Wand2,
  AudioLines,
  Sparkles,
  Activity,
  Users,
  Play,
  Pause
} from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [podcastFile, setPodcastFile] = useState(null);
  const [adScriptFile, setAdScriptFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, detecting_speakers, selecting_speaker, processing, completed, error
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(null);
  const [voiceSimilarity, setVoiceSimilarity] = useState(null);
  const [adPlacement, setAdPlacement] = useState(null);
  
  // Workflow selection state
  const [workflow, setWorkflow] = useState(null); // null = not selected, 'full' = integrate ad, 'ad_only' = just generate ad
  
  // Speaker selection state
  const [speakers, setSpeakers] = useState([]);
  const [selectedSpeaker, setSelectedSpeaker] = useState(null);
  const [playingSpeaker, setPlayingSpeaker] = useState(null);
  const audioRef = useRef(null);

  // Podcast dropzone
  const onDropPodcast = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setPodcastFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps: getPodcastRootProps, getInputProps: getPodcastInputProps, isDragActive: isPodcastDragActive } = useDropzone({
    onDrop: onDropPodcast,
    accept: {
      'audio/*': ['.mp3', '.wav', '.ogg', '.m4a', '.flac'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1
  });

  // Ad script dropzone
  const onDropAdScript = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setAdScriptFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps: getAdScriptRootProps, getInputProps: getAdScriptInputProps, isDragActive: isAdScriptDragActive } = useDropzone({
    onDrop: onDropAdScript,
    accept: {
      'text/*': ['.txt', '.md']
    },
    maxFiles: 1
  });

  // Chunk size: 5MB
  const CHUNK_SIZE = 5 * 1024 * 1024;

  // Upload a single file in chunks
  const uploadFileInChunks = async (file, fileType, jobId, onProgress) => {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    
    // Initialize upload
    const initResponse = await axios.post(`${API_BASE_URL}/api/upload/init`, {
      filename: file.name,
      fileSize: file.size,
      fileType: fileType
    });
    
    const { upload_id } = initResponse.data;
    
    // Upload chunks
    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);
      
      const formData = new FormData();
      formData.append('uploadId', upload_id);
      formData.append('chunkIndex', i);
      formData.append('totalChunks', totalChunks);
      formData.append('chunk', chunk);
      
      await axios.post(`${API_BASE_URL}/api/upload/chunk`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      if (onProgress) {
        onProgress(Math.round(((i + 1) / totalChunks) * 100));
      }
    }
    
    // Complete upload
    const completeResponse = await axios.post(`${API_BASE_URL}/api/upload/complete`, {
      uploadId: upload_id,
      jobId: jobId
    });
    
    return completeResponse.data.file_path;
  };

  // Upload and process files
  const handleProcess = async () => {
    if (!podcastFile || !adScriptFile) {
      setError('Please upload both a podcast file and an ad script.');
      return;
    }

    if (!workflow) {
      setError('Please select a workflow first.');
      return;
    }

    setStatus('uploading');
    setProgress(0);
    setMessage('Uploading files...');
    setError(null);

    try {
      // Generate job ID upfront for chunked uploads
      const jobId = crypto.randomUUID();
      
      // Check if podcast file is large (> 50MB) - use chunked upload
      const usChunkedUpload = podcastFile.size > 50 * 1024 * 1024;
      
      if (usChunkedUpload) {
        setMessage('Uploading podcast (large file, using chunked upload)...');
        
        // Upload podcast in chunks
        const podcastPath = await uploadFileInChunks(
          podcastFile, 
          'podcast', 
          jobId,
          (chunkProgress) => {
            // Podcast is ~90% of upload, script is ~10%
            setProgress(Math.round(chunkProgress * 0.9));
            setMessage(`Uploading podcast... ${Math.round(chunkProgress)}%`);
          }
        );
        
        setMessage('Uploading ad script...');
        
        // Upload ad script (usually small, but use chunks for consistency)
        const adScriptPath = await uploadFileInChunks(
          adScriptFile,
          'adScript',
          jobId,
          (chunkProgress) => {
            setProgress(90 + Math.round(chunkProgress * 0.1));
          }
        );
        
        // Finalize the upload
        console.log('Finalizing chunked upload...');
        await axios.post(`${API_BASE_URL}/api/upload/chunked`, {
          jobId: jobId,
          podcastPath: podcastPath,
          adScriptPath: adScriptPath,
          workflow: workflow
        });
        console.log('Chunked upload finalized, job ID:', jobId);
        
        setJobId(jobId);
      } else {
        // Use regular upload for smaller files
        console.log('Using regular upload for smaller files');
        const formData = new FormData();
        formData.append('podcast', podcastFile);
        formData.append('adScript', adScriptFile);
        formData.append('workflow', workflow);

        const uploadResponse = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(percentCompleted);
          }
        });
        console.log('Regular upload complete, job ID:', uploadResponse.data.job_id);

        setJobId(uploadResponse.data.job_id);
        
        // Detect speakers before processing
        console.log('Calling detectSpeakers for regular upload...');
        await detectSpeakers(uploadResponse.data.job_id);
        return;
      }

      setJobId(jobId);
      
      // For chunked uploads, detect speakers
      console.log('Calling detectSpeakers for chunked upload, job ID:', jobId);
      await detectSpeakers(jobId);
    } catch (err) {
      console.error('Error:', err);
      setStatus('error');
      setError(err.response?.data?.error || 'An error occurred during upload.');
    }
  };

  // Detect speakers in the podcast
  const detectSpeakers = async (id) => {
    try {
      setStatus('detecting_speakers');
      setProgress(10);
      setMessage('Analyzing podcast for multiple speakers...');
      
      console.log('Calling detect-speakers API for job:', id);
      const response = await axios.post(`${API_BASE_URL}/api/detect-speakers/${id}`);
      console.log('Speaker detection response:', response.data);
      
      const { num_speakers, speakers: detectedSpeakers, requires_selection } = response.data;
      
      if (requires_selection && num_speakers > 1) {
        // Multiple speakers detected, show selection UI
        console.log(`Found ${num_speakers} speakers, showing selection UI`);
        setSpeakers(detectedSpeakers);
        setStatus('selecting_speaker');
        setMessage(`Found ${num_speakers} speakers. Please select which voice to use for the ad.`);
      } else {
        // Single speaker, proceed directly
        console.log('Single speaker detected, proceeding to processing');
        setSelectedSpeaker(0);
        await startProcessing(id, 0);
      }
    } catch (err) {
      console.error('Speaker detection error:', err);
      console.error('Error response:', err.response?.data);
      // Show error to user instead of silently proceeding
      setStatus('error');
      setError(`Speaker detection failed: ${err.response?.data?.error || err.message}. Please try again.`);
    }
  };

  // Play speaker sample
  const playSpeakerSample = (speakerId) => {
    if (playingSpeaker === speakerId) {
      // Stop playing
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setPlayingSpeaker(null);
    } else {
      // Stop previous audio
      if (audioRef.current) {
        audioRef.current.pause();
      }
      // Play new sample
      const audio = new Audio(`${API_BASE_URL}/api/speaker-sample/${jobId}/${speakerId}`);
      audio.onended = () => setPlayingSpeaker(null);
      audio.play();
      audioRef.current = audio;
      setPlayingSpeaker(speakerId);
    }
  };

  // Confirm speaker selection and start processing
  const confirmSpeakerSelection = async () => {
    if (selectedSpeaker === null) {
      setError('Please select a speaker.');
      return;
    }
    
    try {
      // Notify backend of selection
      await axios.post(`${API_BASE_URL}/api/select-speaker/${jobId}`, {
        speaker_id: selectedSpeaker
      }, {
        headers: { 'Content-Type': 'application/json' }
      });
      
      // Start processing
      await startProcessing(jobId, selectedSpeaker);
    } catch (err) {
      console.error('Error selecting speaker:', err);
      setError('Failed to select speaker.');
    }
  };

  // Start the main processing
  const startProcessing = async (id, speakerId) => {
    setStatus('processing');
    setProgress(5);
    setMessage('Starting processing...');
    setSpeakers([]);
    
    try {
      await axios.post(`${API_BASE_URL}/api/process/${id}`);
      pollStatus(id);
    } catch (err) {
      console.error('Processing error:', err);
      setStatus('error');
      setError(err.response?.data?.error || 'An error occurred starting processing.');
    }
  };

  // Poll for job status
  const pollStatus = async (id) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/status/${id}`);
      const { status: jobStatus, progress: jobProgress, message: jobMessage, voice_similarity, ad_placement } = response.data;

      setProgress(jobProgress);
      setMessage(jobMessage);

      if (jobStatus === 'completed') {
        setStatus('completed');
        if (voice_similarity) {
          setVoiceSimilarity(voice_similarity);
        }
        if (ad_placement) {
          setAdPlacement(ad_placement);
        }
      } else if (jobStatus === 'error') {
        setStatus('error');
        setError(jobMessage);
      } else if (jobStatus === 'processing') {
        // Continue polling
        setTimeout(() => pollStatus(id), 2000);
      }
    } catch (err) {
      console.error('Status poll error:', err);
      setStatus('error');
      setError('Failed to get processing status.');
    }
  };

  // Download result
  const handleDownload = async () => {
    if (!jobId) return;

    try {
      const response = await axios.get(`${API_BASE_URL}/api/download/${jobId}`, {
        responseType: 'blob'
      });

      // Get filename from content-disposition header or generate one
      const contentDisposition = response.headers['content-disposition'];
      let filename = `integrated_podcast_${jobId}.mp3`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
      setError('Failed to download the processed file.');
    }
  };

  // Reset to start over
  const handleReset = () => {
    // Stop any playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setPodcastFile(null);
    setAdScriptFile(null);
    setJobId(null);
    setStatus('idle');
    setProgress(0);
    setMessage('');
    setError(null);
    setVoiceSimilarity(null);
    setAdPlacement(null);
    setSpeakers([]);
    setSelectedSpeaker(null);
    setPlayingSpeaker(null);
    setWorkflow(null);
  };

  // Get color for similarity score
  const getSimilarityColor = (score) => {
    if (score >= 80) return '#22c55e'; // green
    if (score >= 65) return '#84cc16'; // lime
    if (score >= 50) return '#eab308'; // yellow
    if (score >= 35) return '#f97316'; // orange
    return '#ef4444'; // red
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format time in mm:ss or hh:mm:ss
  const formatTime = (seconds) => {
    if (!seconds && seconds !== 0) return '0:00';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="logo">
            <Sparkles className="logo-icon" />
            <h1>Naarad</h1>
          </div>
          <p className="tagline">AI-Powered Podcast Ad Integration</p>
        </header>

        {/* Main Content */}
        <main className="main">
          {/* Workflow Selection - shown when no workflow is selected */}
          {status === 'idle' && workflow === null && (
            <div className="workflow-selection">
              <h2>What would you like to do?</h2>
              <p className="workflow-subtitle">Choose your workflow to get started</p>
              <div className="workflow-options">
                <div 
                  className="workflow-card"
                  onClick={() => setWorkflow('full')}
                >
                  <div className="workflow-icon">
                    <AudioLines size={48} />
                  </div>
                  <h3>Full Integration</h3>
                  <p>Generate an ad in the podcast voice and seamlessly integrate it into your podcast</p>
                  <ul className="workflow-features">
                    <li>✓ Voice cloning from podcast</li>
                    <li>✓ AI-generated ad audio</li>
                    <li>✓ Automatic ad placement</li>
                    <li>✓ Download integrated podcast</li>
                  </ul>
                </div>
                <div 
                  className="workflow-card"
                  onClick={() => setWorkflow('ad_only')}
                >
                  <div className="workflow-icon">
                    <Mic size={48} />
                  </div>
                  <h3>Generate Ad Only</h3>
                  <p>Clone the podcast voice and generate just the ad audio for manual use</p>
                  <ul className="workflow-features">
                    <li>✓ Voice cloning from podcast</li>
                    <li>✓ AI-generated ad audio</li>
                    <li>✓ Download ad separately</li>
                    <li>✓ Use anywhere you want</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {status === 'idle' && workflow !== null && (
            <>
              {/* Workflow indicator */}
              <div className="workflow-indicator">
                <span className="workflow-badge">
                  {workflow === 'full' ? (
                    <><AudioLines size={16} /> Full Integration</>
                  ) : (
                    <><Mic size={16} /> Ad Only</>
                  )}
                </span>
                <button 
                  className="btn-change-workflow"
                  onClick={() => setWorkflow(null)}
                >
                  Change
                </button>
              </div>

              {/* Upload Section */}
              <div className="upload-section">
                {/* Podcast Upload */}
                <div
                  {...getPodcastRootProps()}
                  className={`dropzone ${isPodcastDragActive ? 'active' : ''} ${podcastFile ? 'has-file' : ''}`}
                >
                  <input {...getPodcastInputProps()} />
                  <div className="dropzone-content">
                    {podcastFile ? (
                      <>
                        <FileAudio className="dropzone-icon success" />
                        <p className="dropzone-filename">{podcastFile.name}</p>
                        <p className="dropzone-filesize">{formatFileSize(podcastFile.size)}</p>
                        <button
                          className="btn-remove"
                          onClick={(e) => {
                            e.stopPropagation();
                            setPodcastFile(null);
                          }}
                        >
                          Remove
                        </button>
                      </>
                    ) : (
                      <>
                        <Upload className="dropzone-icon" />
                        <p className="dropzone-title">Upload Podcast</p>
                        <p className="dropzone-subtitle">
                          Drag & drop your audio/video file here, or click to select
                        </p>
                        <p className="dropzone-formats">
                          Supported: MP3, WAV, OGG, M4A, FLAC, MP4, AVI, MOV, MKV, WebM
                        </p>
                      </>
                    )}
                  </div>
                </div>

                {/* Ad Script Upload */}
                <div
                  {...getAdScriptRootProps()}
                  className={`dropzone ${isAdScriptDragActive ? 'active' : ''} ${adScriptFile ? 'has-file' : ''}`}
                >
                  <input {...getAdScriptInputProps()} />
                  <div className="dropzone-content">
                    {adScriptFile ? (
                      <>
                        <FileText className="dropzone-icon success" />
                        <p className="dropzone-filename">{adScriptFile.name}</p>
                        <p className="dropzone-filesize">{formatFileSize(adScriptFile.size)}</p>
                        <button
                          className="btn-remove"
                          onClick={(e) => {
                            e.stopPropagation();
                            setAdScriptFile(null);
                          }}
                        >
                          Remove
                        </button>
                      </>
                    ) : (
                      <>
                        <FileText className="dropzone-icon" />
                        <p className="dropzone-title">Upload Ad Script</p>
                        <p className="dropzone-subtitle">
                          Drag & drop your ad script here, or click to select
                        </p>
                        <p className="dropzone-formats">
                          Supported: TXT, MD
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="error-message">
                  <XCircle className="error-icon" />
                  <p>{error}</p>
                </div>
              )}

              {/* Process Button */}
              <button
                className="btn-primary"
                onClick={handleProcess}
                disabled={!podcastFile || !adScriptFile}
              >
                <Wand2 className="btn-icon" />
                {workflow === 'ad_only' ? 'Generate Ad Audio' : 'Generate Integrated Podcast'}
              </button>

              {/* Features */}
              <div className="features">
                <div className="feature">
                  <Mic className="feature-icon" />
                  <h3>Voice Cloning</h3>
                  <p>Ad is read in the podcaster's own voice</p>
                </div>
                {workflow === 'full' && (
                  <div className="feature">
                    <AudioLines className="feature-icon" />
                    <h3>Smart Placement</h3>
                    <p>AI finds the perfect spot for your ad</p>
                  </div>
                )}
                <div className="feature">
                  <Sparkles className="feature-icon" />
                  <h3>{workflow === 'ad_only' ? 'High Quality' : 'Smooth Transitions'}</h3>
                  <p>{workflow === 'ad_only' ? 'Professional ad audio output' : 'Natural integration with the content'}</p>
                </div>
              </div>
            </>
          )}

          {(status === 'uploading' || status === 'processing' || status === 'detecting_speakers') && (
            <div className="processing-section">
              <div className="processing-animation">
                <Loader2 className="spinner" />
              </div>
              <h2 className="processing-title">
                {status === 'uploading' ? 'Uploading Files...' : 
                 status === 'detecting_speakers' ? 'Analyzing Speakers...' : 'Processing Your Podcast'}
              </h2>
              <p className="processing-message">{message}</p>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="progress-text">{progress}% Complete</p>

              {status === 'processing' && (
                <div className="processing-steps">
                  <div className={`step ${progress >= 10 ? 'active' : ''} ${progress >= 20 ? 'completed' : ''}`}>
                    <span className="step-number">1</span>
                    <span className="step-text">Analyzing Ad Script</span>
                  </div>
                  <div className={`step ${progress >= 20 ? 'active' : ''} ${progress >= 50 ? 'completed' : ''}`}>
                    <span className="step-number">2</span>
                    <span className="step-text">Transcribing Podcast</span>
                  </div>
                  <div className={`step ${progress >= 50 ? 'active' : ''} ${progress >= 60 ? 'completed' : ''}`}>
                    <span className="step-number">3</span>
                    <span className="step-text">Finding Best Placement</span>
                  </div>
                  <div className={`step ${progress >= 60 ? 'active' : ''} ${progress >= 80 ? 'completed' : ''}`}>
                    <span className="step-number">4</span>
                    <span className="step-text">Generating Ad Audio</span>
                  </div>
                  <div className={`step ${progress >= 80 ? 'active' : ''} ${progress >= 100 ? 'completed' : ''}`}>
                    <span className="step-number">5</span>
                    <span className="step-text">Integrating Ad</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {status === 'selecting_speaker' && (
            <div className="speaker-selection-section">
              <Users className="speaker-icon" />
              <h2 className="speaker-title">Multiple Speakers Detected</h2>
              <p className="speaker-subtitle">
                We found {speakers.length} different voices in your podcast. 
                Please select which voice should be used for the ad.
              </p>
              
              <div className="speakers-grid">
                {speakers.map((speaker) => (
                  <div
                    key={speaker.id}
                    className={`speaker-card ${selectedSpeaker === speaker.id ? 'selected' : ''}`}
                    onClick={() => setSelectedSpeaker(speaker.id)}
                  >
                    <div className="speaker-card-header">
                      <div className="speaker-avatar">
                        <Users className="speaker-avatar-icon" />
                      </div>
                      <span className="speaker-name">Speaker {speaker.id + 1}</span>
                    </div>
                    
                    <div className="speaker-card-body">
                      <p className="speaker-duration">
                        ~{Math.round(speaker.total_speaking_time)}s of speech
                      </p>
                      
                      <button
                        className={`btn-play ${playingSpeaker === speaker.id ? 'playing' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          playSpeakerSample(speaker.id);
                        }}
                      >
                        {playingSpeaker === speaker.id ? (
                          <>
                            <Pause className="play-icon" />
                            Stop Sample
                          </>
                        ) : (
                          <>
                            <Play className="play-icon" />
                            Play Sample
                          </>
                        )}
                      </button>
                    </div>
                    
                    {selectedSpeaker === speaker.id && (
                      <div className="speaker-selected-badge">
                        <CheckCircle2 className="check-icon" />
                        Selected
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {error && (
                <div className="error-message">
                  <XCircle className="error-icon" />
                  <p>{error}</p>
                </div>
              )}

              <div className="speaker-actions">
                <button
                  className="btn-primary"
                  onClick={confirmSpeakerSelection}
                  disabled={selectedSpeaker === null}
                >
                  <Wand2 className="btn-icon" />
                  Continue with Selected Voice
                </button>
                <button className="btn-secondary" onClick={handleReset}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          {status === 'completed' && (
            <div className="completed-section">
              <CheckCircle2 className="completed-icon" />
              <h2 className="completed-title">
                {workflow === 'ad_only' ? 'Your Ad is Ready!' : 'Your Podcast is Ready!'}
              </h2>
              <p className="completed-message">
                {workflow === 'ad_only' 
                  ? 'The ad has been successfully generated in the podcaster\'s voice.'
                  : 'The ad has been successfully integrated into your podcast.'
                }
              </p>
              
              {/* Voice Similarity Score */}
              {voiceSimilarity && (
                <div className="similarity-score-container">
                  <div className="similarity-header">
                    <Activity className="similarity-icon" />
                    <h3>Voice Similarity Score</h3>
                  </div>
                  <div className="similarity-gauge">
                    <div 
                      className="similarity-gauge-fill"
                      style={{ 
                        width: `${voiceSimilarity.score}%`,
                        backgroundColor: getSimilarityColor(voiceSimilarity.score)
                      }}
                    />
                  </div>
                  <div className="similarity-details">
                    <span 
                      className="similarity-score"
                      style={{ color: getSimilarityColor(voiceSimilarity.score) }}
                    >
                      {voiceSimilarity.score}%
                    </span>
                    <span className="similarity-label">{voiceSimilarity.quality_label}</span>
                  </div>
                  <p className="similarity-description">
                    This score measures how similar the generated ad voice is to the original podcast speaker.
                  </p>
                </div>
              )}

              {/* Ad Placement Visualization - Only for full workflow */}
              {adPlacement && workflow === 'full' && (
                <div className="ad-placement-container">
                  <div className="ad-placement-header">
                    <AudioLines className="ad-placement-icon" />
                    <h3>Ad Placement</h3>
                  </div>
                  
                  {/* Timeline visualization */}
                  <div className="timeline-container">
                    <div className="timeline-bar">
                      {/* Before ad section */}
                      <div 
                        className="timeline-section timeline-before"
                        style={{ 
                          width: `${(adPlacement.insertion_point / adPlacement.final_duration) * 100}%` 
                        }}
                      />
                      {/* Ad section */}
                      <div 
                        className="timeline-section timeline-ad"
                        style={{ 
                          width: `${(adPlacement.ad_duration / adPlacement.final_duration) * 100}%` 
                        }}
                      >
                        <span className="timeline-ad-label">AD</span>
                      </div>
                      {/* After ad section */}
                      <div 
                        className="timeline-section timeline-after"
                        style={{ 
                          width: `${((adPlacement.final_duration - adPlacement.ad_end) / adPlacement.final_duration) * 100}%` 
                        }}
                      />
                    </div>
                    
                    {/* Time markers */}
                    <div className="timeline-markers">
                      <span className="timeline-marker">0:00</span>
                      <span className="timeline-marker timeline-marker-ad">
                        {formatTime(adPlacement.insertion_point)}
                      </span>
                      <span className="timeline-marker timeline-marker-end">
                        {formatTime(adPlacement.final_duration)}
                      </span>
                    </div>
                  </div>
                  
                  {/* Placement details */}
                  <div className="ad-placement-details">
                    <div className="placement-stat">
                      <span className="placement-stat-label">Ad Starts At</span>
                      <span className="placement-stat-value">{formatTime(adPlacement.insertion_point)}</span>
                    </div>
                    <div className="placement-stat">
                      <span className="placement-stat-label">Ad Duration</span>
                      <span className="placement-stat-value">{formatTime(adPlacement.ad_duration)}</span>
                    </div>
                    <div className="placement-stat">
                      <span className="placement-stat-label">Total Duration</span>
                      <span className="placement-stat-value">{formatTime(adPlacement.final_duration)}</span>
                    </div>
                  </div>
                  
                  {adPlacement.is_natural_break && (
                    <p className="placement-note">
                      <CheckCircle2 className="note-icon" />
                      Ad placed at a natural pause in the conversation
                    </p>
                  )}
                </div>
              )}
              
              <div className="completed-actions">
                <button className="btn-primary" onClick={handleDownload}>
                  <Download className="btn-icon" />
                  {workflow === 'ad_only' ? 'Download Ad Audio' : 'Download Integrated Podcast'}
                </button>
                <button className="btn-secondary" onClick={handleReset}>
                  {workflow === 'ad_only' ? 'Generate Another Ad' : 'Process Another Podcast'}
                </button>
              </div>
            </div>
          )}

          {status === 'error' && (
            <div className="error-section">
              <XCircle className="error-icon-large" />
              <h2 className="error-title">Processing Failed</h2>
              <p className="error-description">{error}</p>
              <button className="btn-secondary" onClick={handleReset}>
                Try Again
              </button>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="footer">
          <p>Powered by Open Source AI Models from HuggingFace</p>
        </footer>
      </div>
    </div>
  );
}

export default App;

