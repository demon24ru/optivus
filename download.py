import os
from loguru import logger


def download_model(hg_model):
    model_name = hg_model.rsplit('/', 1)[-1]
    model_path = os.path.join("./", "LLM", model_name)

    if not os.path.exists(model_path):
        logger.info(f"Downloading model to: {model_path}")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=hg_model,
                          local_dir=model_path,
                          local_dir_use_symlinks=False)


if __name__ == '__main__':
    # hg_model = 'openai/whisper-base'
    # hg_model = 'openai/whisper-small'
    hg_model = 'openai/whisper-medium'
    download_model(hg_model)