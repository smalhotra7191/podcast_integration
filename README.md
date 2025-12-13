# Naarad - Podcast Ad Integration Platform

An AI-powered web application that seamlessly integrates ads into podcasts using voice cloning and smart placement algorithms.

## 🎯 Features

- **Smart Ad Placement**: AI analyzes your podcast to find the perfect moment to insert the ad
- **Voice Cloning**: Generates the ad audio in the podcaster's own voice
- **Smooth Transitions**: Creates natural transitions into and out of the ad
- **Multi-format Support**: Works with both audio (MP3, WAV, etc.) and video podcasts (MP4, AVI, etc.)
- **Open Source AI**: Uses free, open-source models from HuggingFace

## 🏗️ Architecture

```
├── frontend/                # React web application
│   ├── src/
│   │   ├── App.js          # Main application component
│   │   └── App.css         # Styles
│   └── package.json
│
└── backend/                 # Python Flask API
    ├── app.py              # Flask server
    ├── orchestrator.py     # Pipeline coordinator
    ├── agents/
    │   ├── ad_analyzer.py      # Analyzes ad scripts
    │   ├── podcast_analyzer.py # Transcribes & analyzes podcasts
    │   ├── voice_cloning.py    # Generates ad in podcaster's voice
    │   └── audio_integrator.py # Integrates ad into podcast
    └── requirements.txt
```

## 🤖 AI Agents

### 1. Ad Script Analyzer
- Uses **RoBERTa** for question-answering to extract product info
- Uses **BART** for summarization
- Extracts: product name, problem solved, target audience, keywords, call-to-action

### 2. Podcast Analyzer
- Uses **OpenAI Whisper** (via HuggingFace) for speech-to-text transcription
- Uses **Sentence-BERT** for semantic similarity matching
- Finds the best insertion point based on content relevance

### 3. Voice Cloning Agent
- Uses **Microsoft SpeechT5** for text-to-speech synthesis
- Extracts speaker embeddings from podcast audio
- Generates ad audio that mimics the podcaster's voice

### 4. Audio Integrator
- Uses **pydub** for audio manipulation
- Creates smooth transitions with crossfades
- Handles both audio and video formats

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- FFmpeg (for video processing)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### FFmpeg Installation

**Windows:**
```bash
winget install FFmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

## 📖 How It Works

1. **Upload**: User uploads a podcast file and an ad script
2. **Analyze Ad**: AI extracts key information from the ad script (product, problem, keywords)
3. **Transcribe Podcast**: Whisper model transcribes the podcast audio
4. **Find Placement**: Semantic search finds the most relevant section for ad placement
5. **Clone Voice**: Speaker embedding is extracted and used to generate ad audio
6. **Integrate**: Ad is seamlessly inserted with smooth transitions
7. **Download**: User downloads the integrated podcast

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend folder:

```env
FLASK_ENV=development
FLASK_DEBUG=1
MAX_CONTENT_LENGTH=500000000
```

### Model Selection

The application uses these HuggingFace models by default:
- `openai/whisper-base` - Speech recognition
- `facebook/bart-large-cnn` - Text summarization
- `deepset/roberta-base-squad2` - Question answering
- `microsoft/speecht5_tts` - Text-to-speech
- `all-MiniLM-L6-v2` - Sentence embeddings

You can modify these in the respective agent files.

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload podcast and ad script |
| `/api/process/<job_id>` | POST | Start processing |
| `/api/status/<job_id>` | GET | Get processing status |
| `/api/download/<job_id>` | GET | Download result |
| `/api/analysis/<job_id>` | GET | Get analysis details |

## 🎨 Tech Stack

### Frontend
- React 18
- Axios for API calls
- react-dropzone for file uploads
- Lucide React for icons

### Backend
- Flask (Python web framework)
- Transformers (HuggingFace)
- PyDub (audio processing)
- Librosa (audio analysis)
- Sentence-Transformers (semantic search)

## ⚠️ Notes

- First run will download AI models (~2-3GB)
- GPU recommended for faster processing
- Processing time depends on podcast length
- Voice cloning quality improves with longer speaker samples

## 📄 License

MIT License

## 🙏 Acknowledgments

- HuggingFace for open-source AI models
- OpenAI Whisper team
- Microsoft SpeechT5 team
