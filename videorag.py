import os
import shutil
from multiprocessing import Process, Queue
from loguru import logger

from videorag._llm import *
from videorag.videorag import VideoRAG as VideoRagService, QueryParam


class VideoRAG(Process):
    working_dir = './videorag-workdir'
    prompt = None
    dtype = None
    device = None
    queue = None
    qsio = None
    qres = None

    def __init__(self, qsio=None, queue=None, qres=None):
        super().__init__()
        self.daemon = True

        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('AI_VISIBLE_DEVICES', '0')
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
        os.environ["OPENAI_BASE_URL"] = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
        # set OLLAMA_HOST or pass in host="http://127.0.0.1:11434"
        os.environ["OLLAMA_HOST"] = os.getenv('AI_OLLAMA_HOST', 'http://ollama:11434')
        self.llm_config = LLMConfig(
            # embedding_func_raw=openai_embedding,
            # embedding_model_name="text-embedding-3-small",
            # embedding_dim=1536,
            # embedding_max_token_size=8192,
            # embedding_batch_num=32,
            # embedding_func_max_async=16,
            embedding_func_raw=ollama_embedding,
            embedding_model_name="bge-m3:latest",
            embedding_dim=1024,
            embedding_max_token_size=8192,
            embedding_batch_num=10,
            embedding_func_max_async=8,

            query_better_than_threshold=0.2,

            # LLM (we utilize gpt-4o-mini for all experiments)
            best_model_func_raw=gpt_4o_mini_complete,
            best_model_name="gpt-4o-mini",
            best_model_max_token_size=32768,
            best_model_max_async=16,

            cheap_model_func_raw=gpt_4o_mini_complete,
            cheap_model_name="gpt-4o-mini",
            cheap_model_max_token_size=32768,
            cheap_model_max_async=16
        )

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
        if os.path.exists(self.working_dir):
            self._load_rag()

        logger.info(f"{self.__class__.__name__} start")

        while True:
            data = self.queue.get()

            # Process immediately for pipeline tasks
            self._handle_result(data)

    def _load_rag(self):
        self.videorag = VideoRagService(llm=self.llm_config, working_dir=self.working_dir)
        self.videorag.load_caption_model(debug=False)

    def _clear_rag(self):
        try:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
        except OSError:
            pass
        self.videorag = VideoRagService(llm=self.llm_config, working_dir=self.working_dir)

    def _insert_video(self, audio_transcribe, video_paths, video_split):
        self.videorag.insert_video(
            speech_to_text_store=audio_transcribe,
            video_path_list=video_paths,
            split_video_store=video_split
        )
        self.videorag.load_caption_model(debug=False)

    def _pre_query(self):
        self.videorag_param = QueryParam(mode="videorag")
        self.videorag_param.wo_reference = True

    def _query(self, query):
        return self.videorag.query(query=query, param=self.videorag_param)

    def _handle_result(self, data):
        """Handle the transcription result"""
        # Check if this is a pipeline task
        if 'task_id' in data:
            if 'step' in data and data['step'] == 'split':
                try:
                    self._clear_rag()
                    result = {
                        'result': self.videorag.split_video(video_path_list=data['files'])
                    }
                except Exception as e:
                    logger.warning(f"Error split video to audio: {e}")
                    result = {
                        'error': f"Error split video to audio: {e}"
                    }

                # Report back to ResultQueue
                self.qres.put({
                    'action': 'add_part',
                    'task_id': data['task_id'],
                    'part_name': 'video',
                    **result
                })
            elif 'step' in data and data['step'] == 'query':
                result = {'result': 'OK'}
                video_split = data.get('video_split', [])
                audio_transcribe = data.get('audio_transcribe', [])
                files = data.get('files', [])

                try:
                    self._insert_video(audio_transcribe, files, video_split)
                    for file in files:
                        os.remove(file)
                except Exception as e:
                    result = {'error': f"Error insert RAG videos: {str(e)}"}

                if 'query' in data and 'error' not in result:
                    self._pre_query()
                    dat = []
                    for i, item in enumerate(data['query']):
                        # Regular task, send result to user
                        try:
                            dat.append({'result': self._query(item)})
                        except Exception as e:
                            dat.append({'error': str(e)})
                    result = {'result': dat}
                # Report back to ResultQueue
                self.qres.put({
                    'action': 'add_part',
                    'task_id': data['task_id'],
                    'part_name': 'query',
                    **result
                })
        else:
            if 'query' in data:
                self._pre_query()
                dat = []
                for i, item in enumerate(data['query']):
                    # Regular task, send result to user
                    try:
                        dat.append({'result': self._query(item)})
                    except Exception as e:
                        dat.append({'error': str(e)})
                data['data'] = dat

            self.qsio.put(data)
