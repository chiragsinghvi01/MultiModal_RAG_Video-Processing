# MultiModal RAG Video Processing

A comprehensive multimodal Retrieval-Augmented Generation (RAG) system that processes YouTube videos by extracting audio, converting speech to text, extracting frames as images, and enabling intelligent querying across both text and visual content using advanced AI models.

## 🚀 Features

- **YouTube Video Download**: Download videos directly from YouTube URLs using `yt-dlp`
- **Multimodal Processing**: Extract both audio/text and visual frames from videos
- **Speech-to-Text**: Convert video audio to text using OpenAI Whisper
- **Frame Extraction**: Extract video frames at configurable intervals for visual analysis
- **Vector Database Storage**: Store embeddings in LanceDB for efficient similarity search
- **Multimodal RAG**: Query both text and image content simultaneously
- **AI-Powered QA**: Use OpenAI's GPT-4o model for intelligent responses based on retrieved content
- **Visual Results**: Display relevant video frames alongside text responses

## 🛠️ Tech Stack

### Core Libraries
- **LlamaIndex**: Multimodal RAG framework and vector store integration
- **LanceDB**: High-performance vector database for embeddings storage
- **OpenAI GPT-4o**: Multimodal language model for intelligent responses
- **HuggingFace Embeddings**: BAAI/bge-small-en-v1.5 for text embeddings

### Video/Audio Processing
- **MoviePy**: Video processing, frame extraction, and audio conversion
- **yt-dlp**: YouTube video downloading
- **SpeechRecognition**: Audio-to-text conversion with Whisper integration
- **OpenAI Whisper**: State-of-the-art speech recognition

### Data Processing & Visualization
- **PIL (Pillow)**: Image processing and manipulation
- **Matplotlib**: Visualization and plotting
- **NumPy & PyTorch**: Numerical computing and tensor operations
- **scikit-image**: Advanced image processing

### Additional Tools
- **FFmpeg**: Media file handling and conversion
- **CLIP**: Vision-language understanding
- **Pydub**: Audio manipulation
- **SoundFile**: Audio file I/O

## 📊 Key Components

### Video Processing Pipeline
1. **Download**: Fetch video from YouTube using `yt-dlp`
2. **Frame Extraction**: Extract frames at 0.2 FPS (configurable)
3. **Audio Processing**: Convert video audio to WAV format
4. **Speech Recognition**: Transcribe audio using Whisper
5. **Data Storage**: Save text and organize frames

### RAG System
1. **Embedding Generation**: Create embeddings for text and images
2. **Vector Storage**: Store in separate LanceDB collections
3. **Retrieval**: Find relevant content based on query similarity
4. **Generation**: Use GPT-4o to generate responses with context

### Query Engine
- **Similarity Search**: Find top-k similar text and image content
- **Multimodal Context**: Combine text and visual information
- **AI Response**: Generate comprehensive answers using retrieved context

## ⚠️ Important Notes

- Ensure you have sufficient storage space for video files and extracted frames
- Large videos will generate many frames and require more processing time
- OpenAI API usage will incur costs based on your query volume
- Some videos may have download restrictions or require special handling
