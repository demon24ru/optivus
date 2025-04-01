from multiprocessing import Process, Queue
import os
import time
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from download import download_model
import uuid
import shutil
from pathlib import Path

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
from fish_speech.utils.schema import ServeTTSRequest


class FishSpeech(Process):
    folder_path = os.getenv('FS_PATH', './fs_audio')
    queue = None
    qsio = None

    def __init__(self, qsio=None):
        super().__init__()
        self.daemon = True

        self.queue = Queue()
        self.qsio = qsio
        self.device = "cuda"
        self.half = True
        self.compile = False
        self.max_text_length = 1000
        
        # Create the folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def put(self, data, sid):
        self.queue.put({**data, 'sid': sid})
    
    def initialize_models(self):
        hg_model = os.getenv('AI_FISH_MODEL', 'fishaudio/fish-speech-1.5')
        model_name = hg_model.rsplit('/', 1)[-1]
        model_path = os.path.join("./", "LLM", model_name)

        if not os.path.exists(model_path):
            download_model(hg_model)

        self.llama_checkpoint_path = model_path
        self.decoder_checkpoint_path = os.path.join(model_path, 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth')
        self.decoder_config_name = 'firefly_gan_vq'

        """Initialize TTS models similar to ModelManager in the API server"""
        logger.info("Initializing TTS models...")
        
        # Set precision
        self.precision = torch.half if self.half else torch.bfloat16
        
        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("MPS is available, running on MPS.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")
            
        # Load the TTS models
        self.load_llama_model()
        self.load_decoder_model()
        
        # Initialize the TTS inference engine
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )
        
        # Warm up the models
        self.warm_up()
        logger.info("TTS models initialized successfully.")
    
    def load_llama_model(self):
        """Load the LLAMA model for text-to-semantic conversion"""
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.llama_checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile=self.compile,
        )
        logger.info("LLAMA model loaded.")
    
    def load_decoder_model(self):
        """Load the decoder model for semantic-to-audio conversion"""
        self.decoder_model = load_vqgan_model(
            config_name=self.decoder_config_name,
            checkpoint_path=self.decoder_checkpoint_path,
            device=self.device,
        )
        logger.info("Decoder model loaded.")
    
    def warm_up(self):
        """Warm up the models with a simple inference"""
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        
        # Run a simple inference to warm up the models
        try:
            file, _ = self.process_tts_request(request)
            os.remove(file)
            logger.info("Models warmed up.")
        except Exception as e:
            logger.error(f"Error during model warm-up: {e}")
    
    def process_tts_request(self, request):
        """Process a TTS request and return the generated audio"""
        # Check if the text is too long
        if self.max_text_length > 0 and len(request.text) > self.max_text_length:
            logger.error(f"Text is too long, max length is {self.max_text_length}")
            return None
        
        # Get the sample rate from the decoder model
        sample_rate = self.decoder_model.spec_transform.sample_rate
        
        # Process the TTS request
        audio_data = None
        try:
            start_time = time.time()
            
            # Collect all audio segments
            audio_segments = []
            for result in self.tts_inference_engine.inference(request):
                if result.code == "error":
                    logger.error(f"TTS error: {result.error}")
                    return None
                elif result.code == "final":
                    if isinstance(result.audio, tuple):
                        audio_data = result.audio[1]
                        break
                elif result.code == "segment":
                    if isinstance(result.audio, tuple):
                        audio_segments.append(result.audio[1])
            
            # If we have segments but no final audio, concatenate the segments
            if audio_data is None and len(audio_segments) > 0:
                audio_data = np.concatenate(audio_segments, axis=0)
            
            logger.info(f"TTS processing time: {(time.time() - start_time) * 1000:.2f}ms")
            
            # Save the audio to a file
            if audio_data is not None:
                # Generate a unique filename
                timestamp = int(time.time())
                filename = f"{self.folder_path}/tts_{timestamp}.wav"
                
                # Save the audio to a file
                sf.write(filename, audio_data, sample_rate, format="wav")
                logger.info(f"Audio saved to {filename}")
                
                return filename, audio_data
            
        except Exception as e:
            logger.error(f"Error processing TTS request: {e}")
        
        return None

    def process_reference(self, audio_path, transcription_text):
        """
        Process a reference audio file and its transcription text.
        Save them to the references directory and return the reference_id.
        
        Args:
            audio_path (str): Path to the audio file
            transcription_text (str): Transcription text for the audio file
            
        Returns:
            str: Reference ID (folder name in references directory)
        """
        try:
            # Create a unique reference ID
            reference_id = str(uuid.uuid4())
            
            # Create the reference directory
            reference_dir = Path(self.folder_path) / "references" / reference_id
            reference_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the audio file to the reference directory
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return None
                
            # Get the audio file extension
            extension = audio_path.suffix.lower()
            if extension not in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                logger.error(f"Unsupported audio format: {extension}")
                return None
                
            # Copy the audio file to the reference directory
            dest_audio_path = reference_dir / f"reference{extension}"
            shutil.copy2(audio_path, dest_audio_path)
            
            # Create the transcription file
            transcription_path = dest_audio_path.with_suffix('.lab')
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(transcription_text)
                
            logger.info(f"Reference processed and saved with ID: {reference_id}")
            return reference_id
            
        except Exception as e:
            logger.error(f"Error processing reference: {e}")
            return None

    def run(self):
        # Initialize the models when the process starts
        self.initialize_models()
        
        while True:
            data = self.queue.get()
            
            # Check if this is a reference processing request
            if 'reference_audio' in data and 'reference_text' in data:
                try:
                    # Process the reference audio and text
                    reference_id = self.process_reference(
                        data['reference_audio'], 
                        data['reference_text']
                    )
                    
                    # Send the result back through the queue
                    if self.qsio:
                        self.qsio.put({
                            **data,
                            'reference_id': reference_id,
                            'status': 'success' if reference_id else 'error',
                            'message': 'Reference processed successfully' if reference_id else 'Failed to process reference'
                        })
                
                except Exception as e:
                    logger.error(f"Error in reference processing: {e}")
                    if self.qsio:
                        self.qsio.put({
                            **data,
                            'reference_id': None,
                            'status': 'error',
                            'error': str(e)
                        })
            
            # Check if this is a TTS request
            elif 'text' in data and isinstance(data['text'], str):
                try:
                    # Create a TTS request
                    tts_request = ServeTTSRequest(
                        text=data['text'],
                        references=data.get('references', []),
                        reference_id=data.get('reference_id', None),
                        max_new_tokens=data.get('max_new_tokens', 1024),
                        chunk_length=data.get('chunk_length', 200),
                        top_p=data.get('top_p', 0.7),
                        repetition_penalty=data.get('repetition_penalty', 1.2),
                        temperature=data.get('temperature', 0.7),
                        format="wav",
                        streaming=False
                    )
                    
                    # Process the TTS request
                    result = self.process_tts_request(tts_request)
                    
                    if result:
                        filename, audio_data = result
                        
                        # Send the result back through the queue
                        if self.qsio:
                            self.qsio.put({
                                **data,
                                'fish': filename,
                                'audio_data': audio_data.tolist() if isinstance(audio_data, np.ndarray) else None,
                            })
                    else:
                        # Send an error message
                        if self.qsio:
                            self.qsio.put({
                                **data,
                                'fish': '',
                                'error': 'Failed to generate audio'
                            })
                
                except Exception as e:
                    logger.error(f"Error in TTS processing: {e}")
                    if self.qsio:
                        self.qsio.put({
                            **data,
                            'fish': '',
                            'error': str(e)
                        })
            else:
                # Handle other types of requests
                if self.qsio:
                    self.qsio.put({
                        **data,
                        'fish': '' # TODO: Implement other fish speech features
                    })
