import React, { useState, useCallback } from 'react';
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
  Sparkles
} from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [podcastFile, setPodcastFile] = useState(null);
  const [adScriptFile, setAdScriptFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, completed, error
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(null);

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

  // Upload and process files
  const handleProcess = async () => {
    if (!podcastFile || !adScriptFile) {
      setError('Please upload both a podcast file and an ad script.');
      return;
    }

    setStatus('uploading');
    setProgress(0);
    setMessage('Uploading files...');
    setError(null);

    try {
      // Upload files
      const formData = new FormData();
      formData.append('podcast', podcastFile);
      formData.append('adScript', adScriptFile);

      const uploadResponse = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });

      const { job_id } = uploadResponse.data;
      setJobId(job_id);
      setStatus('processing');
      setProgress(5);
      setMessage('Starting processing...');

      // Start processing
      await axios.post(`${API_BASE_URL}/api/process/${job_id}`);

      // Poll for status
      pollStatus(job_id);
    } catch (err) {
      console.error('Error:', err);
      setStatus('error');
      setError(err.response?.data?.error || 'An error occurred during upload.');
    }
  };

  // Poll for job status
  const pollStatus = async (id) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/status/${id}`);
      const { status: jobStatus, progress: jobProgress, message: jobMessage } = response.data;

      setProgress(jobProgress);
      setMessage(jobMessage);

      if (jobStatus === 'completed') {
        setStatus('completed');
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
    setPodcastFile(null);
    setAdScriptFile(null);
    setJobId(null);
    setStatus('idle');
    setProgress(0);
    setMessage('');
    setError(null);
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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
          {status === 'idle' && (
            <>
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
                Generate Integrated Podcast
              </button>

              {/* Features */}
              <div className="features">
                <div className="feature">
                  <Mic className="feature-icon" />
                  <h3>Voice Cloning</h3>
                  <p>Ad is read in the podcaster's own voice</p>
                </div>
                <div className="feature">
                  <AudioLines className="feature-icon" />
                  <h3>Smart Placement</h3>
                  <p>AI finds the perfect spot for your ad</p>
                </div>
                <div className="feature">
                  <Sparkles className="feature-icon" />
                  <h3>Smooth Transitions</h3>
                  <p>Natural integration with the content</p>
                </div>
              </div>
            </>
          )}

          {(status === 'uploading' || status === 'processing') && (
            <div className="processing-section">
              <div className="processing-animation">
                <Loader2 className="spinner" />
              </div>
              <h2 className="processing-title">
                {status === 'uploading' ? 'Uploading Files...' : 'Processing Your Podcast'}
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

          {status === 'completed' && (
            <div className="completed-section">
              <CheckCircle2 className="completed-icon" />
              <h2 className="completed-title">Your Podcast is Ready!</h2>
              <p className="completed-message">
                The ad has been successfully integrated into your podcast.
              </p>
              <div className="completed-actions">
                <button className="btn-primary" onClick={handleDownload}>
                  <Download className="btn-icon" />
                  Download Integrated Podcast
                </button>
                <button className="btn-secondary" onClick={handleReset}>
                  Process Another Podcast
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
