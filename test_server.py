import socketio
import asyncio


async def main():
    async with socketio.AsyncSimpleClient() as sio:
        await sio.connect('http://localhost:5000')
        await sio.emit('message', {
            # 'type': 'dlp',
            # 'data_type': 'file',
            # 'type': 'video',
            'type': 'whisp',
            'file': './test_video/youtube__ChZ0Q2Q58E_audio.mp3',
            # 'v_conv': True,
            'id': 123456,
            # 'urls': [
            #     'https://www.youtube.com/shorts/5J8Ej0tv90k',
            #     'https://www.youtube.com/shorts/gutZtqv1ghU',
            #     'https://www.youtube.com/shorts/YegGIxkGUSY',
                # 'https://www.youtube.com/shorts/oHR-zDVJhuE',
                # 'https://www.tiktok.com/@shoelover99/video/7441259401496792363'
            # ],

            # 'type': 'fish',
            # 'reference_audio': './fs_audio/tts_1743416383.wav',
            # 'reference_text': 'Раз, два, три, четыре, пять. Вышел зайчик погулять. И не просто вышел сам, а пришел парам пам пам.',
            # 'text': 'Шел по улице человек без лица и на прохожих такой у-у-у-у-у!!!',
            # 'reference_id': 'd70bd88c-67d7-4951-a394-b80d07aba851',
        })
        while True:
            dat = await sio.receive()
            print(dat)
            if len(dat) > 1 and dat[1]['data'] is not None:
                break


if __name__ == '__main__':
    asyncio.run(main())