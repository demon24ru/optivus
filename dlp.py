import os
from time import sleep
from multiprocessing import Process, Queue
from yt_dlp import YoutubeDL
from loguru import logger


class Dlp(Process):
    folder_path = os.getenv('PATH_VIDEO', './test_video')
    queue = None
    qsio = None
    qres = None
    ydl_opts = {}

    def __init__(self, qsio=None, queue=None, qres=None):
        super().__init__()
        self.daemon = True

        self.queue = queue or Queue()
        self.qsio = qsio
        self.qres = qres
        
        # Create the folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            
        # Set up yt-dlp options
        self.ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp3]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.folder_path, '%(epoch)s_%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }

    def put(self, data):
        """Add a task to the queue"""
        self.queue.put(data)

    def put_with_sid(self, data, sid):
        """Add a task to the queue with sid"""
        self.queue.put({**data, 'sid': sid})

    def run(self):
        while True:
            data = self.queue.get()
            tp = data.get('data_type', 'info')
            result = []
            
            # Process each URL (should be only one in the new approach)
            for url in data.get('urls', []):
                try:
                    if tp == 'info':
                        result.append({**self.extract_file(url), 'url': url})
                    elif tp == 'file':
                        result.append({**self.extract_file(url, download=True), 'url': url})
                except Exception as e:
                    result.append({'error': str(e), 'url': url})

            # Check if this is a pipeline task
            if 'task_id' in data:
                # This is a task from ResultQueue, report back
                self.qres.put({
                    'action': 'download_complete',
                    'task_id': data['task_id'],
                    'part_name': 'dlp',
                    'item_index': data.get('item_index', 0),  # Include the item index
                    'result': result
                })
            else:
                # Regular task, send result to user
                self.qsio.put({
                    **data,
                    'data': result
                })

    def extract_file(self, URL, download=False, ppt=2):
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                extr = ydl.extract_info(URL, download=download)
                info = ydl.sanitize_info(extr)
                return {
                    'uploader': info.get('uploader'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'dislike_count': info.get('dislike_count'),
                    'repost_count': info.get('repost_count'),
                    'average_rating': info.get('average_rating'),
                    'comment_count': info.get('comment_count'),
                    'age_limit': info.get('age_limit'),
                    'media_type': info.get('media_type'),
                    'fulltitle': info.get('fulltitle'),
                    'tags': info.get('tags'),
                    'categories': info.get('categories'),
                    'description': info.get('description'),
                    'file': os.path.join(self.folder_path, os.path.basename(ydl.prepare_filename(info))) if download else None
                }
        except Exception as e:
            logger.error(f"Error extracting file: {URL}, {str(e)}, {ppt} attempts left")
            if ppt > 0:
                ppt -= 1
                sleep(1)
                return self.extract_file(URL, download=download, ppt=ppt)
            else:
                raise e
