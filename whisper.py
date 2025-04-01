import os
from multiprocessing import Process, Queue
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from download import download_model
import torch
import ffmpeg
from loguru import logger


class Whisper(Process):
    pipe = None
    queue = None
    qsio = None
    qres = None

    def __init__(self, qsio=None, queue=None, qres=None):
        super().__init__()
        self.daemon = True

        self.queue = queue or Queue()
        self.qsio = qsio
        self.qres = qres

    def put(self, data):
        """Add a task to the queue"""
        self.queue.put(data)

    def put_with_sid(self, data, sid):
        """Add a task to the queue with sid"""
        self.queue.put({**data, 'sid': sid})

    def load_model(self):
        hg_model = os.getenv('AI_WHISPER_MODEL', 'openai/whisper-base')
        model_name = hg_model.rsplit('/', 1)[-1]
        model_path = os.path.join("./", "LLM", model_name)

        if not os.path.exists(model_path):
            download_model(hg_model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float32
        logger.info(f"{self.__class__.__name__} device => {device}")

        # Load Whisper model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_path)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def run(self):
        self.load_model()

        buf_num = int(os.getenv('AI_WHISPER_BATCH_WORKER', '4'))
        data_buf = []
        while True:
            data = self.queue.get()

            # Check if this is a batch task with 'data' field
            if 'data' in data and isinstance(data['data'], list):
                # If there are pending tasks in data_buf, process them first
                if data_buf:
                    res = self.transcribe(data_buf)
                    for i in range(len(data_buf)):
                        self._handle_result(data_buf[i], res[i])
                    data_buf = []
                
                # Process batch task
                batch_data = data['data']
                batch_results = []
                
                # Process the batch in chunks of buf_num
                for i in range(0, len(batch_data), buf_num):
                    chunk = batch_data[i:i+buf_num]
                    
                    # Process the chunk
                    res = self.transcribe(chunk)
                    batch_results.extend(res)
                
                # Handle the batch results
                self._handle_batch_result(data, batch_results)
            else:
                # Buffer for batch processing
                data_buf.append(data)
                if not self.queue.empty() and len(data_buf) < buf_num:
                    continue

                res = self.transcribe(data_buf)
                for i in range(len(data_buf)):
                    self._handle_result(data_buf[i], res[i])
                data_buf = []

    def _handle_result(self, data, result):
        """Handle a single result"""
        # Check if this is a pipeline task
        if 'task_id' in data:
            # Report back to ResultQueue
            self.qres.put({
                'action': 'add_part',
                'task_id': data['task_id'],
                'part_name': 'audio',
                'item_index': data.get('item_index', 0),  # Include the item index if present
                'result': result
            })
        else:
            # Regular task, send result to user
            self.qsio.put({
                **data,
                'transcribe_audio': result,
            })
    
    def _handle_batch_result(self, data, batch_results):
        """Handle a batch of results"""
        # Regular task, send result to user
        self.qsio.put({
            **data,
            'data': [
                {**data['data'][i], 'transcribe_audio': batch_results[i]} 
                    for i in range(len(data['data']))
                ]
            })

    def transcribe(self, data_buf):
        audioPaths = self._pre_transcribe(data_buf)
        files = [v['file'] for v in audioPaths if v['file'] is not None]
        # Transcribe audio using Whisper
        try:
            audioTranscription = self.pipe(files, batch_size=len(files), return_timestamps=True)
        except Exception as e:
            logger.warning(f"Error transcribing audio: {e}")
            audioTranscription = [{'error': str(e)}] * len(files)
        result = []
        j = 0
        for t in audioPaths:
            if t['error'] is not None:
                result.append({
                    'error': t['error'],
                    'text': '',
                    'chunks': []
                })
                continue
            result.append({
                'error': audioTranscription[j].get('error', None),
                'text': audioTranscription[j].get('text', ''),
                'chunks': [
                    {
                        "start": int(item["timestamp"][0]),
                        "end": int(item["timestamp"][1] if item["timestamp"][1] is not None else item["timestamp"][0]),
                        "text": item["text"]
                    } for item in audioTranscription[j].get('chunks', [])
                ]
            })
            j += 1
        self._post_transcribe(files)
        return result

    def _pre_transcribe(self, data_buf):
        video_types = ['mp4']
        audio_types = ['mp3', 'wav']
        audioPaths = []
        for data in data_buf:
            if 'file' in data:
                file = data['file']
                if isinstance(file, str):
                    if file.split(".")[-1] in audio_types:
                        audioPaths.append({
                            "file": file,
                            "error": None
                            })
                    elif file.split(".")[-1] in video_types:
                        videoPath = file
                        audioPath = file.rsplit(".", 1)[0] + ".wav"

                        try:
                            ffmpeg.input(videoPath).output(audioPath, q='0', map='a', y=None, loglevel='quiet').run()
                        except Exception as e:
                            logger.warning(f"Error extracting audio: {e}")
                            audioPaths.append({
                                "file": None,
                                "error": f"Error extracting audio: {e}"
                            })
                            continue
                        audioPaths.append({
                            "file": audioPath,
                            "error": None
                        })
        
        return audioPaths

    def _post_transcribe(self, audioPaths):
        for file in audioPaths:
            os.remove(file)
