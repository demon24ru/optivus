from aiohttp import web
import socketio
from multiprocessing import Queue
from queue import Empty
from whisper import Whisper
from videorag import VideoRAG
from fish_sp import FishSpeech
from dlp import Dlp
from result_queue import ResultQueue
from loguru import logger


sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

qsio = Queue()
qdlp = Queue()
qwhisp = Queue()
qflrnc = Queue()
qres = Queue()

# Initialize the components
whisp_cls = Whisper(qsio, qwhisp, qres)
videorag_cls = VideoRAG(qsio, qflrnc, qres)
dlp_cls = Dlp(qsio, qdlp, qres)
result_queue = ResultQueue(qsio, qres, qdlp, qwhisp, qflrnc)  # Pass all necessary queues
fish_speech_cls = FishSpeech(qsio)


async def response(status='200', data=None, sid=None):
    await sio.send({
        'status': status,
        'data': data
    }, room=sid)


async def background_task():
    while True:
        try:
            data = qsio.get(False)
        except Empty:
            await sio.sleep(0.5)
            continue

        await response(data=data, sid=data['sid'])


@sio.event
async def message(sid, data):
    tp = data.get('type', 'whisp')
    
    # Check if this is a pipeline task
    if data.get('v_conv', False):
        # Send all pipeline tasks to ResultQueue
        result_queue.put(data, sid)
    elif tp == 'fish':
        fish_speech_cls.put(data, sid)
    elif tp == 'whisp':
        whisp_cls.put_with_sid(data, sid)
    elif tp == 'video':
        # Mark as pipeline task and send to ResultQueue
        data['v_conv'] = True
        result_queue.put(data, sid)
    elif tp == 'dlp':
        dlp_cls.put_with_sid(data, sid)


@sio.event
async def disconnect_request(sid):
    await sio.disconnect(sid)


@sio.event
async def connect(sid, environ):
    logger.info(f'Client connect: {sid}')


@sio.event
def disconnect(sid, reason):
    logger.info(f'Client disconnect: {sid}, reason: {reason}')


async def init_app():
    sio.start_background_task(background_task)
    result_queue.start()
    whisp_cls.start()
    videorag_cls.start()
    dlp_cls.start()
    fish_speech_cls.start()
    return app


if __name__ == '__main__':
    web.run_app(init_app(), host='0.0.0.0', port=5000)