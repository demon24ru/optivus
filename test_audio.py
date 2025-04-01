import os
import time
import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import ffmpeg


if __name__ == '__main__':
    hg_model = "openai/whisper-small"
    model_name = hg_model.rsplit('/', 1)[-1]
    model_path = os.path.join("./", "LLM", model_name)

    if not os.path.exists(model_path):
        print(f'No folder {model_path}!!! First run the download script!')
        quit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
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

    folder_path = "./test_video"
    if not os.path.exists(folder_path):
        print(f'No folder {folder_path}!!!')
        quit(1)

    if len(os.listdir(folder_path)) > 1:
        start_time = time.time()
        result = run_example([os.path.join(folder_path, item) for item in os.listdir(folder_path)])
        elapsed_time = time.time() - start_time
        print(f'{str(datetime.timedelta(seconds=elapsed_time))} seconds to complete.')
        print(result)
        quit(0)

    for filename in os.listdir(folder_path):
        video_types = ['mp4', 'mp3', 'wav']
        audio_types = ['mp3', 'wav']
        if filename.split(".")[-1] in video_types:
            if filename.split(".")[-1] in audio_types:
                audioPath = os.path.join(folder_path, filename)
            else:
                videoPath = os.path.join(folder_path, filename)
                audioPath = os.path.join(folder_path, filename.split(".")[0] + ".wav")

                ffmpeg.input(videoPath).output(audioPath, q='0', map='a', y=None, loglevel='quiet').run()

            if not os.path.exists(audioPath):
                print("Error: Could not extract audio from the video.")
                quit(1)

            start_time = time.time()
            result = run_example(audioPath)
            elapsed_time = time.time() - start_time
            print(f'{filename} {str(datetime.timedelta(seconds=elapsed_time))} seconds to complete.')
            for r in result:
                # print(f'{str(datetime.timedelta(seconds=r["start"]))}\n{r["text"]}')
                print(f'{r["text"]}')