import os
import time
import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import ffmpeg

PATH_FOLDER = "./test_lightrag"
MODEL = "openai/whisper-medium"

if __name__ == '__main__':
    model_name = MODEL.rsplit('/', 1)[-1]
    model_path = os.path.join("./", "LLM", model_name)

    if not os.path.exists(model_path):
        print(f'No folder {model_path}!!! First run the download script!')
        quit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("device =>", device)
    # Load Whisper model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Function to run the model on an example
    def run_example(audioPath):
        # Transcribe audio using Whisper
        if type(audioPath) is list:
            audioTranscription = pipe(audioPath, batch_size=len(audioPath), return_timestamps=True)
            return [[
                {
                    "start": int(item["timestamp"][0]),
                    "end": int(item["timestamp"][1] if item["timestamp"][1] is not None else item["timestamp"][0]),
                    "text": item["text"]
                } for item in t['chunks']
            ] for t in audioTranscription]
        else:
            audioTranscription = pipe(audioPath, return_timestamps=True)

            return [
                {
                    "start": int(item["timestamp"][0]),
                    "end": int(item["timestamp"][1] if item["timestamp"][1] is not None else item["timestamp"][0]),
                    "text": item["text"]
                } for item in audioTranscription['chunks']
            ]


    if not os.path.exists(PATH_FOLDER):
        print(f'No folder {PATH_FOLDER}!!!')
        quit(1)

    # if len(os.listdir(PATH_FOLDER)) > 1:
    #     start_time = time.time()
    #     result = run_example([os.path.join(PATH_FOLDER, item) for item in os.listdir(PATH_FOLDER)])
    #     elapsed_time = time.time() - start_time
    #     print(f'{str(datetime.timedelta(seconds=elapsed_time))} seconds to complete.')
    #     print(result)
    #     quit(0)

    def transcribe(file, folder):
        audio_types = ['mp3', 'wav']
        support_types = ['mp4'] + audio_types
        filename = file.split(".")[0]
        extension = file.split(".")[-1]
        if extension in support_types:
            if extension in audio_types:
                audioPath = os.path.join(folder, file)
            else:
                audioPath = os.path.join(folder, f"{filename}.wav")

                ffmpeg.input(os.path.join(folder, file)).output(audioPath, q='0', map='a', y=None, loglevel='quiet').run()

            if not os.path.exists(audioPath):
                print("Error: Could not extract audio from the video.")
                quit(1)

            start_time = time.time()
            print(f'{audioPath} transcribe...')
            result = run_example(audioPath)
            elapsed_time = time.time() - start_time
            print(f'{file} {str(datetime.timedelta(seconds=elapsed_time))} seconds to complete.')

            with open(os.path.join(folder, f"{filename}.txt"), "w", encoding='utf-8') as f:
                for r in result:
                    # print(f'{str(datetime.timedelta(seconds=r["start"]))}\n{r["text"]}')
                    f.write(f'{r["text"]}\n')
        else:
            print(f'{os.path.join(folder, file)} Error: not support!')

    def worker(path):
        for elem in os.listdir(path):
            path_el = os.path.join(path, elem)
            if os.path.isfile(path_el):
                transcribe(elem, path)
            elif os.path.isdir(path_el):
                worker(path_el)

    worker(PATH_FOLDER)
