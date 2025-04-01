# Optivus Server

![Optivus Server](cover.webp)

## Overview

Optivus Server is a multifunctional multimedia processing server that provides APIs for working with audio, video, and text. The server is built on a microservices architecture and uses WebSocket for communication with clients.

## Core Components

The server consists of the following main components:

1. **Whisper** — Component for audio transcription
2. **Florence** — Component for image and video analysis
3. **FishSpeech** — Component for text-to-speech synthesis (TTS)
4. **DLP** — Component for downloading videos from various sources
5. **ResultQueue** — Component for managing pipeline processing of tasks

## Installation and Setup

### Requirements
- Python 3.8 or higher
- Dependencies installed from requirements.txt
- CUDA-compatible GPU (optional, for performance acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/demon24ru/optivus.git
cd optivus

# Install dependencies
pip install -r requirements.txt
```

### Starting the Server

```bash
python server.py
```

By default, the server runs on port 5000 and is accessible at `http://0.0.0.0:5000`.

## API and Commands

The server uses WebSocket for data exchange. Below are the main types of requests and their parameters.

### 1. Audio Transcription (Whisper)

**Request Type:** `whisp`

**Parameters:**
- `file` — Path to an audio file or video file for audio extraction

**Request Example:**
```json
{
  "type": "whisp",
  "file": "/path/to/audio.mp3"
}
```

**Response Example:**
```json
{
  "transcribe_audio": {
    "text": "Full transcription text",
    "chunks": [
      {
        "start": 0,
        "end": 5,
        "text": "Text fragment"
      }
    ]
  }
}
```

### 2. Image and Video Analysis (Florence2)

**Request Type:** `flrnc`

**Parameters:**
- `file` — Path to an image or video file

**Request Example:**
```json
{
  "type": "flrnc",
  "file": "/path/to/image.jpg"
}
```

**Response Example for Images:**
```json
{
  "transcribe_img": {
    "text": "Image description"
  }
}
```

**Response Example for Videos:**
```json
{
  "transcribe_video": [
    {
      "start": 0,
      "end": 5,
      "text": "Description of the scene in the video"
    }
  ]
}
```

### 3. Speech Synthesis (FishSpeech)

**Request Type:** `fish`

**Parameters:**
- `text` — Text for speech synthesis
- `references` (optional) — List of references to audio files for voice cloning
- `reference_id` (optional) — Identifier of a saved voice
- `reference_audio` (optional) — Path to an audio file for voice cloning
- `reference_text` (optional) — Text corresponding to the audio file for voice cloning

**Request Example for Speech Synthesis:**
```json
{
  "type": "fish",
  "text": "Text for speech synthesis"
}
```

**Request Example for Saving a Voice Reference:**
```json
{
  "type": "fish",
  "reference_audio": "/path/to/voice.wav",
  "reference_text": "Text that is spoken in the audio file"
}
```

**Response Example for Speech Synthesis:**
```json
{
  "fish": "/path/to/generated/tts_1234567890.wav",
  "audio_data": [...]
}
```

**Response Example for Saving a Reference:**
```json
{
  "reference_id": "unique-reference-id",
  "status": "success",
  "message": "Reference processed successfully"
}
```

### 4. Video Download (DLP)

**Request Type:** `dlp`

**Parameters:**
- `urls` — List of URLs to download
- `data_type` — Data type (`info` for getting information, `file` for downloading the file)

**Request Example:**
```json
{
  "type": "dlp",
  "urls": ["https://www.youtube.com/watch?v=example"],
  "data_type": "file"
}
```

**Response Example:**
```json
{
  "data": [
    {
      "uploader": "Channel Name",
      "duration": 120,
      "view_count": 10000,
      "like_count": 500,
      "fulltitle": "Video Title",
      "description": "Video Description",
      "file": "/path/to/downloaded/video.mp4",
      "url": "https://www.youtube.com/watch?v=example"
    }
  ]
}
```

### 5. Pipeline Video Processing

**Request Type:** `video` or any request with the parameter `v_conv: true`

**Parameters:**
- `file` — Path to a video file or
- `urls` — List of URLs to download and subsequently process

**Request Example:**
```json
{
  "type": "video",
  "file": "/path/to/video.mp4"
}
```

**Response Example:**
```json
{
  "data": [
    {
      "transcribe_audio": {
        "text": "Full transcription text",
        "chunks": [...]
      },
      "transcribe_video": [
        {
          "start": 0,
          "end": 5,
          "text": "Description of the scene in the video"
        }
      ]
    }
  ]
}
```

## Environment Variables

The server supports the following environment variables:

- `AI_WHISPER_MODEL` — Model for audio transcription (default: "openai/whisper-base")
- `AI_FLORENCE_MODEL` — Model for image analysis (default: "microsoft/Florence-2-base")
- `AI_FISH_MODEL` — Model for speech synthesis (default: "fishaudio/fish-speech-1.5")
- `AI_WHISPER_BATCH_WORKER` — Number of parallel tasks for Whisper (default: 4)
- `PATH_VIDEO` — Path for saving downloaded videos (default: "./test_video")
- `FS_PATH` — Path for saving generated audio files (default: "./fs_audio")

## Usage Examples

### Example Using Python and socketio

```python
import socketio

# Create client
sio = socketio.Client()

# Response handler
@sio.on('message')
def on_message(data):
    print("Response received:", data)

# Connect to server
sio.connect('http://localhost:5000')

# Send audio transcription request
sio.emit('message', {
    'type': 'whisp',
    'file': '/path/to/audio.mp3'
})

# Send speech synthesis request
sio.emit('message', {
    'type': 'fish',
    'text': 'Hello, this is a test message for speech synthesis.'
})

# Wait for responses
sio.wait()
```

### Example Using JavaScript and socket.io

```javascript
const io = require('socket.io-client');

// Connect to server
const socket = io('http://localhost:5000');

// Response handler
socket.on('message', (data) => {
  console.log('Response received:', data);
});

// Send audio transcription request
socket.emit('message', {
  type: 'whisp',
  file: '/path/to/audio.mp3'
});

// Send speech synthesis request
socket.emit('message', {
  type: 'fish',
  text: 'Hello, this is a test message for speech synthesis.'
});
```

## Server Architecture

Optivus Server uses an asynchronous architecture based on WebSocket and multiprocessing for parallel task execution:

1. **WebSocket Server** (server.py) — Accepts requests from clients and distributes them to the appropriate handlers
2. **Handler Processes** — Each component (Whisper, Florence, FishSpeech, DLP) runs in a separate process
3. **Message Queues** — Queues are used for data exchange between processes
4. **ResultQueue** — Special component for managing pipeline processing, where results from one process are passed to another

## Scaling

The server can be scaled in the following ways:

1. **Vertical Scaling** — Increasing server resources (CPU, GPU, RAM)
2. **Horizontal Scaling** — Running multiple server instances behind a load balancer
3. **Parallelism Configuration** — Adjusting the number of parallel tasks through environment variables

## Troubleshooting

### Common Problems and Solutions

1. **"CUDA out of memory" Error**
   - Reduce the size of processed data
   - Use smaller models
   - Increase video memory

2. **"File not found" Error**
   - Check the correctness of specified file paths
   - Make sure the server has access rights to the specified directories

3. **Slow Video Processing**
   - Reduce video resolution and duration
   - Use more powerful hardware
   - Configure parallelism parameters

## License

[MIT License](LICENSE)
