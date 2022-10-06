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
from text_to_image import generate_images
from text_to_image import image_grid


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Generate images from text prompts.',
    )
    parser.add_argument(
        '-p', '--prompts', type=str,
        required=True, nargs='+',
        help='Text prompts for image generation.',
    )

    parser.add_argument(
        '-H', '--height', type=int, default=512,
        help='Height of the generated image.',
    )
    parser.add_argument(
        '-w', '--width', type=int, default=512,
        help='Width of the generated image.',
    )

    parser.add_argument(
        '-s', '--steps', type=int, default=50,
        help='Number of inference steps.',
    )
    parser.add_argument(
        '-g', '--guidance-scale', type=float, default=0.5,
        help='How much prefix affects the output.',
    )
    parser.add_argument(
        '-t', '--train-step', type=int, default=1000,
        help='Scheduler train steps.',
    )

    parser.add_argument(
        '-d', '--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for inference.',
    )

    parser.add_argument(
        '--pretrained-text', type=str, default='openai/clip-vit-base-patch32',
        help='Pretrained text model from HuggingFace Hub.',
    )
    parser.add_argument(
        '--pretrained-image', type=str, default='CompVis/stable-diffusion-v1-4',
        help='Pretrained image model from HuggingFace Hub.',
    )

    args = parser.parse_args()

    # Managing devices.
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print('Warning: MPS is not available. Using CPU instead.')
        args.device = 'cpu'

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not available. Using CPU instead.')
        args.device = 'cpu'

    print(args)
    images = generate_images(
        prompts=args.prompts,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    image_grid(images, rows=1, cols=len(images)).show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
