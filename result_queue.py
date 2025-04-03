import time
import os
from multiprocessing import Process, Queue, Lock
import ffmpeg
from loguru import logger


class ResultQueue(Process):
    def __init__(self, qsio=None, qres=None, qdlp=None, qwhisp=None, qflrnc=None):
        super().__init__()
        self.daemon = True
        self.queue = qres or Queue()
        self.qsio = qsio
        self.qdlp = qdlp
        self.qwhisp = qwhisp
        self.qflrnc = qflrnc
        self.tasks = {}
        self.lock = Lock()
        
    def put(self, data, sid):
        """Entry point for all tasks that need pipeline processing"""
        self.queue.put({**data, 'sid': sid})
        
    def register_task(self, task_id, data, sid, expected_parts=None):
        """Register a new task with the queue"""
        with self.lock:
            self.tasks[task_id] = {
                'original_data': data,
                'parts': {},
                'expected_parts': expected_parts or ['audio', 'video', 'dlp'],
                'complete': False,
                'sid': sid,
                'timestamp': time.time(),
                'processed_count': 0,  # Track how many items have been processed
                'total_count': 0,      # Total number of items to process
                'results': []          # Store results in order
            }
            
    def add_part(self, task_id, part_name, result, item_index=None):
        """Add a completed part to a task"""
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Warning: Task {task_id} not found")
                return False
                
            task = self.tasks[task_id]
            
            # If this is a batch task with multiple items
            if item_index is not None:
                # Ensure the parts dictionary has an entry for this item
                if item_index not in task['parts']:
                    task['parts'][item_index] = {}
                
                # Add the result for this part
                task['parts'][item_index][part_name] = result
                
                # Check if all expected parts for this item are complete
                if all(part in task['parts'][item_index] for part in task['expected_parts']):
                    # This item is complete, merge its results
                    merged_item = self._merge_item_results(task, item_index)
                    
                    # Add to results in the correct position
                    while len(task['results']) <= item_index:
                        task['results'].append(None)
                    task['results'][item_index] = merged_item
                    
                    # Increment the processed count
                    task['processed_count'] += 1
                    
                    # Check if all items are complete
                    if task['processed_count'] >= task['total_count']:
                        task['complete'] = True
                        return True
            else:
                # Single item task
                task['parts'][part_name] = result
                
                # Check if all expected parts are complete
                if all(part in task['parts'] for part in task['expected_parts']):
                    task['complete'] = True
                    return True
                    
            return False
            
    def run(self):
        while True:
            data = self.queue.get()
            
            # Check if this is a new task or a result from a handler
            if 'action' not in data:
                # This is a new task from the user
                if data.get('v_conv', False):
                    # This is a pipeline task
                    if 'urls' in data:
                        # Task with URLs needs to be downloaded one by one
                        self._handle_download_task(data)
                    elif 'file' in data:
                        # Task with a file can be processed directly
                        self._handle_file_task(data)
                    else:
                        # Invalid task
                        logger.warning(f"Warning: Invalid task format: {data}")
                else:
                    # Not a pipeline task, pass it through
                    self.qsio.put(data)
            else:
                # This is a result from a handler
                action = data.get('action')
                # A part of a task has been completed
                task_id = data.get('task_id')
                part_name = data.get('part_name')
                result = data.get('result')
                item_index = data.get('item_index')
                
                if action == 'download_complete':
                    # A file has been downloaded, process it immediately
                    self._handle_single_download_complete(data)
                    self.add_part(task_id, part_name, result[0], item_index)
                elif action == 'add_part':
                    # A part of a task has been completed
                    if self.add_part(task_id, part_name, result, item_index):
                        # All parts are complete, send the final result to the user
                        self._handle_task_complete(task_id)
    
    def _handle_download_task(self, data):
        """Handle a task that requires downloading files one by one"""
        # Generate a unique task ID
        task_id = f"{data['sid']}_{int(time.time())}"
        
        # Get the list of URLs
        urls = data.get('urls', [])
        
        # Register the task
        self.register_task(
            task_id, 
            data, 
            data['sid'], 
            ['audio', 'video', 'dlp']
        )
        
        # Set the total count of items to process
        with self.lock:
            self.tasks[task_id]['total_count'] = len(urls)
        
        # Send the first URL for download
        if urls:
            self._download_next_url(task_id, urls, 0)
    
    def _download_next_url(self, task_id, urls, index):
        """Download the next URL in the list"""
        if index >= len(urls):
            return  # All URLs have been processed
            
        # Create a download task for a single URL
        download_task = {
            'task_id': task_id,
            'item_index': index,
            'urls': [urls[index]],
            'data_type': 'file'
        }
        
        # Send to downloader
        self.qdlp.put(download_task)
    
    def _handle_single_download_complete(self, data):
        """Handle the completion of a single file download"""
        task_id = data['task_id']
        item_index = data.get('item_index', 0)
        download_result = data['result'][0] if data['result'] else None
        
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Warning: Task {task_id} not found")
                return
                
            task = self.tasks[task_id]
            original_data = task['original_data']
            
            # Check if download was successful
            if download_result:
                if 'error' in download_result:
                    # Handle download error
                    error_msg = download_result['error']
                    logger.warning(f"Download error: {error_msg}")
                    
                    self.add_part(task_id, 'dlp', {
                        'error': f"Download failed: {error_msg}"
                    }, item_index)
                    # Add error results to both audio and video parts
                    self.add_part(task_id, 'audio', {
                        'text': '',
                        'chunks': [],
                        'error': f"Download failed: {error_msg}"
                    }, item_index)
                    self.add_part(task_id, 'video', {
                        'error': f"Download failed: {error_msg}"
                    }, item_index)
                    
                    # Increment processed count to maintain proper tracking
                    with self.lock:
                        task['processed_count'] += 1
                        
                        # Check if all items are complete
                        if task['processed_count'] >= task['total_count']:
                            task['complete'] = True
                            self._handle_task_complete(task_id)
                elif 'file' in download_result and download_result['file'] is not None:
                    # Process the file if download was successful
                    self._process_file(download_result, task_id, item_index)
            
            # Download the next URL
            urls = original_data.get('urls', [])
            next_index = item_index + 1
            if next_index < len(urls):
                self._download_next_url(task_id, urls, next_index)
    
    def _process_file(self, file_data, task_id, item_index):
        """Process a single file for audio and video transcription"""
        file_path = file_data['file']
        
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.warning(f"Warning: File {file_path} does not exist")
            return

        # Extract audio from video
        audio_path = file_path.rsplit(".", 1)[0] + ".wav"
        is_not_audio_error = True
        try:
            ffmpeg.input(file_path).output(audio_path, q='0', map='a', y=None, loglevel='quiet').run()
        except Exception as e:
            logger.warning(f"Error extracting audio: {e}")
            is_not_audio_error = False
            self.add_part(task_id, 'audio', {
                    'error': str(e),
                    'text': '',
                    'chunks': []
                }, item_index)

        # Send to both processors in parallel
        if is_not_audio_error:
            self.qwhisp.put({
                'file': audio_path,
                'task_id': task_id,
                'item_index': item_index
            })

        self.qflrnc.put({
            'file': file_path,
            'task_id': task_id,
            'item_index': item_index
        })
    
    def _handle_file_task(self, data):
        """Handle a task that already has a file"""
        # Generate a unique task ID
        task_id = f"{data['sid']}_{int(time.time())}"
        
        # Register the task
        self.register_task(
            task_id, 
            data, 
            data['sid'], 
            ['audio', 'video']
        )
        
        # Set the total count to 1 (single file)
        with self.lock:
            self.tasks[task_id]['total_count'] = 1
        
        # Process the file
        file_data = {'file': data['file']}
        self._process_file(file_data, task_id, 0)
    
    def _handle_task_complete(self, task_id):
        """Handle the completion of a task"""
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Warning: Task {task_id} not found")
                return
                
            task = self.tasks[task_id]
            
            # Send the result to the user
            self.qsio.put({
                **task['original_data'],
                'data': task['results'],
                'sid': task['sid']
            })
            
            # Clean up
            del self.tasks[task_id]
    
    def _merge_item_results(self, task, item_index):
        """Merge the results for a single item"""
        parts = task['parts'][item_index]
        
        # Create a merged result
        merged_item = {}
        
        # Add audio and video results
        if 'audio' in parts:
            merged_item['transcribe_audio'] = parts['audio']
        if 'video' in parts:
            merged_item['transcribe_video'] = parts['video']
        if 'dlp' in parts:
            merged_item['info'] = parts['dlp']
            
        return merged_item