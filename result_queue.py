import time
import os
from multiprocessing import Process, Queue, Lock
from loguru import logger


class ResultQueue(Process):
    def __init__(self, qsio=None, qres=None, qdlp=None, qwhisp=None, qvideo=None):
        super().__init__()
        self.daemon = True
        self.queue = qres or Queue()
        self.qsio = qsio
        self.qdlp = qdlp
        self.qwhisp = qwhisp
        self.qvideo = qvideo
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
                'expected_parts': expected_parts or ['dlp', 'video', 'audio', 'query'],
                'complete': False,
                'sid': sid,
                'results': {}          # Store results in order
            }
            
    def add_part(self, task_id, part_name, result, index_number=None, error=None):
        """Add a completed part to a task"""
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Warning: Task {task_id} not found")
                return False
                
            task = self.tasks[task_id]
            
            # If this is a batch task with multiple items
            if index_number is not None:
                # Ensure the parts dictionary has an entry for this item
                if part_name not in task['parts']:
                    task['parts'][part_name] = []
                
                # Add the result for this part
                task['parts'][part_name] += [result]
                
                # Check if all expected parts for this item are complete
                if all(part in task['parts'] for part in task['expected_parts']):
                    # Check if all items are complete
                    if len(task['parts'][part_name]) == index_number:
                        # This item is complete, merge its results
                        task['results'] = self._merge_item_results(task)
                        task['complete'] = True
                        return True
            else:
                if error is not None:
                    task['results'] = {'error': error}
                    task['complete'] = True
                    return True

                # Single item task
                task['parts'][part_name] = result
                
                # Check if all expected parts are complete
                if all(part in task['parts'] for part in task['expected_parts']):
                    # This item is complete, merge its results
                    task['results'] = self._merge_item_results(task)
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
                    elif 'files' in data:
                        # Task with files paths
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
                error = data.get('error')
                
                if action == 'download_complete':
                    # A file has been downloaded, process it immediately
                    self._handle_single_download_complete(data)
                elif action == 'transcribe_audio':
                    # A video file has been split, process it immediately
                    self._handle_transcribe_audio_complete(data)
                elif action == 'add_part':
                    # A part of a task has been completed
                    if self.add_part(task_id, part_name, result, error=error):
                        # All parts are complete, send the final result to the user
                        self._handle_task_complete(task_id)
                    else:
                        self._process_file(task_id)
    
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
            ['dlp', 'video', 'audio', 'query']
        )
        
        # Send the first URL for download
        if urls:
            self._download_next_url(task_id, urls, 0)

    def _handle_file_task(self, data):
        """Handle a task that already has a file"""
        # Generate a unique task ID
        task_id = f"{data['sid']}_{int(time.time())}"

        # Register the task
        self.register_task(
            task_id,
            data,
            data['sid'],
            ['video', 'audio', 'query']
        )

        files = data.get('files', [])

        # Process the file
        if files:
            self._process_file(task_id)
    
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

    def _transcribe_next_file(self, task_id, files, index):
        """Transcribe the next audio file in the list"""
        if index >= len(files):
            return  # All URLs have been processed

        # Create a download task for a single URL
        download_task = {
            'task_id': task_id,
            'item_index': index,
            'data': files[index]['audio_files']
        }

        # Send to downloader
        self.qwhisp.put(download_task)
    
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
            urls = original_data.get('urls', [])
            
            # Check if download was successful
            if download_result:
                if 'error' in download_result:
                    # Handle download error
                    error_msg = download_result['error']
                    logger.warning(f"Download error: {error_msg}")
                    
                    self.add_part(task_id, 'dlp', {
                        'error': f"Download failed: {error_msg}"
                    }, len(urls))
                elif 'file' in download_result and download_result['file'] is not None:
                    # Process the file if download was successful
                    file_path = download_result['file']
                    if not os.path.exists(file_path):
                        logger.warning(f"Warning: File {file_path} does not exist")
                        return self._download_next_url(task_id, urls, item_index)

                    if 'files' not in original_data:
                        original_data['files'] = [file_path]
                    else:
                        original_data['files'] += [file_path]

                    self.add_part(task_id, 'dlp', download_result, len(urls))

            next_index = item_index + 1
            if next_index < len(urls):
                # Download the next URL
                self._download_next_url(task_id, urls, next_index)
            else:
                self._process_file(task_id)

    def _handle_transcribe_audio_complete(self, data):
        """Handle the completion of a single video file split"""
        task_id = data['task_id']
        item_index = data.get('item_index', 0)
        transcribe_result = data['result'] if data['result'] else None

        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Warning: Task {task_id} not found")
                return

            task = self.tasks[task_id]
            original_data = task['parts']['video']

            # Check if download was successful
            if transcribe_result:
                if 'error' in transcribe_result:
                    # Handle download error
                    error_msg = transcribe_result['error']
                    logger.warning(f"Transcribe error: {error_msg}")

                    self.add_part(task_id, 'audio', {
                        'error': f"Transcribe failed: {error_msg}"
                    }, len(original_data))
                else:
                    # Process the audio file if transcribe was successful
                    self.add_part(task_id, 'audio', transcribe_result, len(original_data))

            next_index = item_index + 1
            if next_index < len(original_data):
                # Transcribe the next audio file
                self._transcribe_next_file(task_id, original_data, next_index)
            else:
                self._process_file(task_id)
    
    def _process_file(self, task_id):
        """Process main"""
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Warning: Task {task_id} not found")
                return False

            task = self.tasks[task_id]
            original_data = task['original_data']

            part_name = ''
            for part in task['expected_parts']:
                if part not in task['parts']:
                    part_name = part
                    break

            if part_name == 'video':
                self.qvideo.put({
                    'files': original_data.get('files', []),
                    'step': 'split',
                    'task_id': task_id
                })
            elif part_name == 'audio':
                result = task['parts']['video']
                self._transcribe_next_file(task_id, result, 0)
            elif part_name == 'query':
                audio_transcribe = task['parts']['audio']
                video_split = task['parts']['video']
                self.qvideo.put({
                    'files': original_data.get('files', []),
                    'video_split': video_split,
                    'audio_transcribe': audio_transcribe,
                    'query': original_data.get('query', []),
                    'step': 'query',
                    'task_id': task_id
                })
    
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
    
    def _merge_item_results(self, task):
        """Merge the results for a single item"""
        parts = task['parts']
        
        # Create a merged result
        merged_item = {}
        
        # Add audio and video results
        if 'query' in parts:
            merged_item['query'] = parts['query']
        if 'dlp' in parts:
            merged_item['info'] = parts['dlp']
            
        return merged_item