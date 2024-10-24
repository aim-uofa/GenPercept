pip install -U "huggingface_hub[cli]"
HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download stabilityai/stable-diffusion-2-1 pretrained_weights/stable-diffusion-2-1 --repo-type model