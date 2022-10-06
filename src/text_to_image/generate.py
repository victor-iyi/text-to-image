import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch import autocast  # mypy: allow-implicit-reexport


def generate(prompt: str) -> Image.Image:
    """Generate image from text prompt.

    Args:
        prompt (str): Image text description.

    Returns:
        Image.Image - A Pillow Image object.
    """
    # if isinstance(prompt, str):
    #     prompt = [prompt]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16,  # pylint: disable=no-member
        use_auth_token=True,
    )
    pipe = pipe.to(device)

    with autocast(device):
        image: Image.Image = pipe(prompt)['sample'][0]

    return image
