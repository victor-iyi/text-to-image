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
from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel
from PIL import Image
from torch import autocast  # mypy: allow-implicit-reexport
from torch import nn
from tqdm import tqdm
from transformers import CLIPTextModel
from transformers import CLIPTokenizer


# Pretrained models.
# PRETRAINED_IMAGE_MODEL = 'CompVis/stable-diffusion-v1-4'
# PRETRAINED_TEXT_MODEL = 'openai/clip-vit-large-patch14'
# SCHEDULER_TRAIN_STEPS = 1000


def get_text_embeddings(
    text: str | list[str],
    prefix: str | list[str] = '',
    tokenizer: nn.Module | None = None,
    text_encoder: nn.Module | None = None,
    **kwargs: dict[str, str | int],
) -> torch.Tensor:
    """Get text embeddings from text prompt.

    Args:
        text (str | list[str]): Text prompt.
        prefix (str | list[str], optional): Prefix to add to text prompt.
            Defaults to ''.
        tokenizer (nn.Module, optional) - Pre-trained text tokenizer.
            Defaults to `transformers.CLIPTokenizer`.
        text_encoder (nn.Model, optional) - Pre-trained text encoder.
            Defaults to `transformers.CLIPTextModel`.

    Keyword Args:
        device (str): Device type. Defaults to 'cpu'.
        train_step (int): Scheduler train step. Defaults to 1000.
        pretrained_image_model (str): Pretrained image model from
            hugging face hub.

    Returns:
        torch.Tensor: Text embeddings.
    """
    # Keywrod arguments.
    device = kwargs.get('device')
    cache_dir = kwargs.get('cache_dir')
    pretrained_text_model = kwargs.get('pretrained_text_model')

    if isinstance(text, str):
        text = [text]

    if not tokenizer:
        # Load the tokenizer to tokenize the text prompt..
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_text_model,
            cache_dir=cache_dir,
        )

    # Tokenize the text prompt.
    tokens = tokenizer(
        text, padding='max_length',  # padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt',
    )

    if not text_encoder:
        # Load the encoder to encode the text prompt into embeddings.
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_text_model,
            cache_dir=cache_dir,
        )
        text_encoder = text_encoder.to(device)

    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0]

    if isinstance(prefix, str):
        prefix = [prefix] * len(text)

    # Embed the prefix into the text embeddings.
    prefix_tokens = tokenizer(
        prefix, padding='max_length',  # padding='max_length',
        max_length=tokenizer.model_max_length,
        return_tensors='pt',
    )
    with torch.no_grad():
        prefix_embeddings = text_encoder(
            prefix_tokens.input_ids.to(device),
        )[0]

    # Concatenate the prefix and text embeddings.
    text_embeddings = torch.cat([prefix_embeddings, embeddings])

    return text_embeddings


def produce_latents(
    text_embeddings: torch.Tensor,
    height: int = 512, width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    latents: torch.Tensor | None = None,
    img_model: nn.Module | None = None,
    scheduler: LMSDiscreteScheduler | None = None,
    **kwargs: dict[str, str | int],
) -> torch.Tensor:
    """Produce latent space from text embeddings.

    Args:
        text_embeddings (torch.Tensor): Text embeddings.
        height (int, optional): Height of latent image. Defaults to 512.
        width (int, optional): Width of latent image. Defaults to 512.
        num_inference_steps (int, optional): Number of inference steps.
            Defaults to 50.
        guidance_scale (float, optional): How much guidance should text prefix
            affect the output. Defaults to 7.5.
        latents (torch.Tensor | None, optional): Latent embedding.
            Defaults to None.
        img_model (nn.Module | None, optional): Image model to predict noise
            (UNet model). Defaults to None.
        scheduler (LMSDiscreteScheduler | None, optional): Scheduler for
            inference. Defaults to None.

    Keyword Args:
        device (str): Device type. Defaults to 'cpu'.
        train_step (int): Scheduler train step. Defaults to 1000.
        pretrained_image_model (str): Pretrained image model from
            hugging face hub.

    Returns:
        torch.Tensor: Latent embeddings representing the text prompt as well
            as image noise.
    """

    # Keyword arguments.
    device = kwargs.get('device')
    train_step = kwargs.get('train_step')
    cache_dir = kwargs.get('cache_dir')
    pretrained_image_model = kwargs.get('pretrained_image_model')

    if img_model is None:
        # Load the UNet model for generating latents.
        img_model = UNet2DConditionModel.from_pretrained(
            pretrained_image_model,
            cache_dir=cache_dir,
            subfolder='unet',
            use_auth_token=True,
        )

    if latents is None:
        # Generate random latent noise.
        latents = torch.randn((
            text_embeddings.shape[0] // 2,
            img_model.in_channels,
            height // 8, width // 8,
        ))
    latents = latents.to(device)

    if scheduler is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=train_step,
        )
    scheduler.set_timesteps(num_inference_steps)
    latents *= scheduler.sigmas[0]

    with autocast(device):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # Expand the latents if we are doing classifier-free guidance
            # to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input /= ((sigma ** 2 + 1) ** 0.5)

            # Predict the noise residual.
            with torch.no_grad():
                noise_pred = img_model(
                    latent_model_input, t,
                    encoder_hidden_states=text_embeddings,
                )['sample']

            # Perform guidance
            noise_pred_prefix, noise_pred_text = noise_pred.chunk(2)
            noise_pred = (
                noise_pred_prefix + guidance_scale *
                (noise_pred_text - noise_pred_prefix)
            )

            # Compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)['prev_sample']

    return latents


def decode_img_latents(
    latents: torch.Tensor,
    decoder: AutoencoderKL | None = None,
    **kwargs: dict[str, str],
) -> list[Image.Image]:
    """Decode latent embeddings into images.

    Args:
        latents (torch.Tensor): Latent embeddings.
        decoder (AutoencoderKL | None, optional): Pre-trained
            decoder to decode latent embeddings. Defaults to None.

    Keyword Args:
        device (str): Device type. Defaults to 'cpu'.
        pretrained_image_model (str): Pre-trained model from hugging face hub.

    Returns:
        list[Image.Image]: List of decoded images as a PIL image.
    """
    device = kwargs.get('device')
    cache_dir = kwargs.get('cache_dir')
    pretrained_image_model = kwargs.get('pretrained_image_model')

    latents = 1 / 0.18215 * latents

    if decoder is None:
        decoder = AutoencoderKL.from_pretrained(
            pretrained_image_model,
            cache_dir=cache_dir,
            subfolder='decoder',
            use_auth_token=True,
        )
        decoder = decoder.to(device)

    with torch.no_grad():
        imgs = decoder.decode(latents)['sample']

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')

    pil_imgs = [Image.fromarray(img) for img in imgs]
    return pil_imgs
