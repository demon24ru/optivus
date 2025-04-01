import os
from multiprocessing import Process, Queue
from transformers import AutoModelForCausalLM, AutoProcessor
from download import download_model
from PIL import Image
import torch
import cv2
from loguru import logger


class Florence(Process):
    prompt = None
    dtype = None
    device = None
    queue = None
    qsio = None
    qres = None

    def __init__(self, qsio=None, queue=None, qres=None):
        super().__init__()
        self.daemon = True

        self.prompt = os.getenv('AI_FLORENCE_PROMPT', '<MORE_DETAILED_CAPTION>')
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[os.getenv('AI_FLORENCE_FLOAT', 'fp16')]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue = queue or Queue()
        self.qsio = qsio
        self.qres = qres

    def put(self, data):
        """Add a task to the queue"""
        self.queue.put(data)

    def put_with_sid(self, data, sid):
        """Add a task to the queue with sid"""
        self.queue.put({**data, 'sid': sid})

    def run(self):
        hg_model = os.getenv('AI_FLORENCE_MODEL', 'microsoft/Florence-2-base')
        model_name = hg_model.rsplit('/', 1)[-1]
        model_path = os.path.join("./", "LLM", model_name)

        if not os.path.exists(model_path):
            download_model(hg_model)

        logger.info(f"{self.__class__.__name__} device => {self.device}")

        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='sdpa', device_map=self.device,
                                                     torch_dtype=self.dtype, trust_remote_code=True).to(self.device)

        # Load the processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, device_map=self.device)

        while True:
            data = self.queue.get()

            # Process immediately for pipeline tasks
            result = self.transcribe(model, processor, data)
            self._handle_result(data, result)

    def _handle_result(self, data, result):
        """Handle the transcription result"""
        # Check if this is a pipeline task
        if 'task_id' in data:
            # Report back to ResultQueue
            self.qres.put({
                'action': 'add_part',
                'task_id': data['task_id'],
                'part_name': 'video',
                'item_index': data.get('item_index'),  # Include the item index if present
                'result': result
            })
        else:
            if 'data' in data:
                dat = []
                for i, item in enumerate(data['data']):
                    # Regular task, send result to user
                    if isinstance(result[i], list):
                        if len(result[i]) > 1:
                            dat.append({
                                **item,
                                'transcribe_video': result[i],
                            })
                        else:
                            dat.append({
                                **item,
                                'transcribe_video': result[i][0],
                            })
                    else:
                        dat.append({
                            **item,
                            'transcribe_img': result[i],
                        })
                data['data'] = dat
                self.qsio.put(data)
            else:
                # Regular task, send result to user
                if isinstance(result, list):
                    if len(result) > 1:
                        self.qsio.put({
                            **data,
                            'transcribe_video': result,
                        })
                    else:
                        self.qsio.put({
                            **data,
                            'transcribe_video': result[0],
                        })
                else:
                    self.qsio.put({
                        **data,
                        'transcribe_img': result,
                    })

    def worker(self, model, processor, image):
        # Transcribe image use Florence2
        inputs = processor(text=self.prompt, images=image, return_tensors="pt", do_rescale=False).to(self.dtype).to(self.device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=True,
            early_stopping=False,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return processor.post_process_generation(
            generated_text, task=self.prompt, image_size=(image.width, image.height))[self.prompt]

    def transcribe(self, model, processor, data):
        video_types = ['mp4']
        image_types = ['png', 'jpg', 'jpeg']
        result = None
        
        # Handle batch data
        if 'data' in data and isinstance(data['data'], list):
            batch_results = []
            for item in data['data']:
                if 'file' in item:
                    file = item['file']
                    if item['file'].split(".")[-1] in video_types:
                        # Process video
                        batch_results.append(self._process_video(model, processor, file))
                    elif item['file'].split(".")[-1] in image_types:
                        # Process image
                        batch_results.append(self._process_image(model, processor, file))
            return batch_results
        
        # Handle single file
        if 'file' in data:
            file = data['file']
            if data['file'].split(".")[-1] in video_types:
                result = self._process_video(model, processor, file)
            elif data['file'].split(".")[-1] in image_types:
                result = self._process_image(model, processor, file)

        return result

    def _process_video(self, model, processor, file):
        # Open the video file
        video = cv2.VideoCapture(file)
        # Check if the video file is opened
        if not video.isOpened():
            logger.error(f"Error: Could not open video file {file}")
            return []

        # Get video properties
        fps = int(video.get(cv2.CAP_PROP_FPS))  # Original FPS of the video
        frameSkip = fps  # Skip frames based on desired processing FPS 1sec
        frameCount = 0
        chunks = []
        old_text = None
        try:
            # Process each frame of the video
            while True:
                # Skip frames to achieve 1 FPS
                video.set(cv2.CAP_PROP_POS_FRAMES, frameCount * frameSkip)
                ret, cv2_im = video.read()
                frameCount += 1
                if not ret:
                    break  # Break the loop if no frames are left

                cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                text = self.worker(model, processor, Image.fromarray(cv2_im))
                if old_text is not None and old_text == text:
                    chunks[len(chunks)-1]['end'] = frameCount
                    continue
                old_text = text
                chunks.append({
                    'start': frameCount-1,
                    'end': frameCount,
                    "text": text
                })
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            chunks = [{'error': str(e)}]
            
        # Close the video file
        video.release()
        
        # Clean up
        os.remove(file)
        
        return chunks

    def _process_image(self, model, processor, file):
        try:
            image = Image.open(file)
            # Ensure the image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

            result = {"text": self.worker(model, processor, image)}
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            result = {'error': str(e)}
        
        # Clean up
        os.remove(file)
        
        return result
