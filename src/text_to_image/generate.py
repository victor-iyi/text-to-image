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
import argparse

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from text_to_image.models import decode_img_latents
from text_to_image.models import get_text_embeddings
from text_to_image.models import produce_latents
from torch import autocast  # mypy: allow-implicit-reexport


def generate(
    args: argparse.Namespace,
) -> Image.Image:
    """Generate image from text prompt.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        Image.Image - A Pillow Image object.
    """
    if isinstance(args.prompts, str):
        args.prompts = [args.prompts]

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_image_model, revision='fp16',
        torch_dtype=torch.float16,  # pylint: disable=no-member
        use_auth_token=True,
    )
    pipe = pipe.to(args.device)

    with autocast(args.device):
        image: Image.Image = pipe(
            args.prompts,
            num_inference_steps=args.steps,
        )['sample'][0]

    return image


def generate_images(
    args: argparse.Namespace,
    latents: torch.Tensor | None = None,
) -> list[Image.Image]:
    """Generate image(s) from text prompt(s).

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        latents (torch.Tensor | None, optional): Image latent embeddings. Defaults to None.

    Returns:
        list[Image.Image]: Image or list of generated PIL images.
    """
    kwargs = vars(args)

    # Prompts -> text embeddings.
    text_embeddings = get_text_embeddings(args.prompts, **kwargs)

    # Text embeddings -> image latents.
    latents = produce_latents(
        text_embeddings,
        height=args.height, width=args.width,
        latents=latents,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        **kwargs,
    )

    # Image latents -> Images.
    imgs = decode_img_latents(latents, **kwargs)

    return imgs
