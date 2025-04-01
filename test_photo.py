import os
import time
import datetime
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

    folder_path = "./test_img"
    if not os.path.exists(folder_path):
        print(f'No folder {folder_path}!!!')
        quit(1)

    prompt = "<MORE_DETAILED_CAPTION>"
    for filename in os.listdir(folder_path):
        image_types = ['png', 'jpg', 'jpeg']
        if filename.split(".")[-1] in image_types:
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path)
            # Ensure the image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

            start_time = time.time()
            result = run_example(prompt, image)
            elapsed_time = time.time() - start_time
            print(f'{filename} {str(datetime.timedelta(seconds=elapsed_time))}\n{result[prompt]}')
