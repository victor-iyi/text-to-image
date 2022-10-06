<!--
 Copyright 2022 Victor I. Afolabi

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Text to Image

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/victor-iyi/text-to-image/main.svg)](https://results.pre-commit.ci/latest/github/victor-iyi/text-to-image/main)

Generate realistic images from text prompt.

If you're using Apple Silicon, you can take advantage of the Apple's
Metal Performance Shader (MPS) to use the fast M-series chip.

To use this, make sure to install the nightly release of PyTorch as it is not
yet avvailable in the sable release (but will be added in future releases).

```sh
# MPS acceleration is available on MacOS 12.3+
$ pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## HuggingFace 🤗 Hub

To log in to huggingface hub using [`huggingface-cli`].
See [hub documentation] for more installation details.

[`huggingface-cli`]: https://huggingface.co/docs/huggingface_hub/quick-start
[hub documentation]: https://huggingface.co/docs/hub/index

 <!-- markdownlint-disable MD014 commands-show-output -->
```sh
$ huggingface-cli login
```

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (Apache)

This project is opened under the [Apache License 2.0][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[original repository]: https://github.com/victor-iyi/text-to-image
[issues]: https://github.com/victor-iyi/text-to-image/issues
[license]: ./LICENSE
