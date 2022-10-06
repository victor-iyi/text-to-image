# Copyright 2022 Victor I. Afolabi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from text_to_image.models import decode_img_latents
from text_to_image.models import get_text_embeddings
from text_to_image.models import produce_latents
from torch import autocast  # mypy: allow-implicit-reexport


def generate(
    prompts: str | list[str],
    steps: int = 50,
) -> Image.Image:
    """Generate image from text prompt.

    Args:
        prompt (str | list[str]): Image text description.

    Returns:
        Image.Image - A Pillow Image object.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16,  # pylint: disable=no-member
        use_auth_token=True,
    )
    pipe = pipe.to(device)

    with autocast(device):
        image: Image.Image = pipe(
            prompts,
            num_inference_steps=steps,
        )['sample'][0]

    return image


def generate_images(
    prompts: str | list[str],
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    latents: torch.Tensor | None = None,
) -> list[Image.Image]:
    """Generate image(s) from text prompt(s).

    Args:
        prompts (str | list[str]): Text prompt(s).
        height (int, optional): Generated image height. Defaults to 512.
        width (int, optional): Generated image width. Defaults to 512.
        num_inference_steps (int, optional): Number of inference steps. Defaults to 50.
        guidance_scale (float, optional): How much to prefix affects output. Defaults to 7.5.
        latents (torch.Tensor | None, optional): Image latent embeddings. Defaults to None.

    Returns:
        list[Image.Image]: Image or list of generated PIL images.
    """
    # Prompts -> text embeddings.
    text_embeddings = get_text_embeddings(prompts)

    # Text embeddings -> image latents.
    latents = produce_latents(
        text_embeddings,
        height=height, width=width,
        latents=latents,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # Image latents -> Images.
    imgs = decode_img_latents(latents)

    return imgs
