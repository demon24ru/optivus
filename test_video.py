import os
import time
import datetime
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image


if __name__ == '__main__':
    # hg_model = 'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
    hg_model = 'MiaoshouAI/Florence-2-base-PromptGen-v2.0'
    model_name = hg_model.rsplit('/', 1)[-1]
    model_path = os.path.join("./", "LLM", model_name)
    attention = 'sdpa'
    precision = 'fp16'
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    if not os.path.exists(model_path):
        print(f'No folder {model_path}!!! First run the download script!')
        quit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =>", device)

    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device,
                                                     torch_dtype=dtype, trust_remote_code=True).to(device)

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, device_map=device)

    # Function to run the model on an example
    def run_example(task_prompt, image):
        inputs = processor(text=task_prompt, images=image, return_tensors="pt", do_rescale=False).to(dtype).to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=True,
            early_stopping=False,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return processor.post_process_generation(generated_text, task=task_prompt,
                                                          image_size=(image.width, image.height))

    folder_path = "./test_video"
    if not os.path.exists(folder_path):
        print(f'No folder {folder_path}!!!')
        quit(1)

    prompt = "<MORE_DETAILED_CAPTION>"
    for filename in os.listdir(folder_path):
        video_types = ['mp4']
        if filename.split(".")[-1] in video_types:
            video_path = os.path.join(folder_path, filename)
            # Open the video file
            video = cv2.VideoCapture(video_path)
            # Check if the video file is opened
            if not video.isOpened():
                print("Error: Could not open video file.")
                quit(1)

            # Get video properties
            fps = int(video.get(cv2.CAP_PROP_FPS))  # Original FPS of the video
            frameSkip = fps  # Skip frames based on desired processing FPS 1sec
            frameCount = 0

            # Process each frame of the video
            while True:
                # Skip frames to achieve 1 FPS
                video.set(cv2.CAP_PROP_POS_FRAMES, frameCount * frameSkip)
                ret, cv2_im = video.read()
                frameCount += 1
                if not ret:
                    break  # Break the loop if no frames are left

                cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(cv2_im)
                image.save(os.path.join(folder_path, f'{frameCount}.jpg'))

                start_time = time.time()
                result = run_example(prompt, image)
                elapsed_time = time.time() - start_time
                print(f'{filename} {frameCount-1}s {str(datetime.timedelta(seconds=elapsed_time))}\n{result[prompt]}')
