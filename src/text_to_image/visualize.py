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
from PIL import Image


def image_grid(
        imgs: list[Image.Image], rows: int, cols: int,
) -> Image.Image:
    """Create image grid from list of PIL Images.

    Args:
        imgs (list[Image.Image]): List of PIL images.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        Image.Image - A single PIL Image organized into grids.
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
