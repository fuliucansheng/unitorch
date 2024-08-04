<div align="Center"> 

![unitorch](https://raw.githubusercontent.com/fuliucansheng/unitorch/master/unitorch.png)


[Documentation](https://fuliucansheng.github.io/unitorch) •
[Installation Instructions](https://fuliucansheng.github.io/unitorch/installation/) •
[Reporting Issues](https://github.com/fuliucansheng/unitorch/issues/new?assignees=&labels=&template=bug-report.yml)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unitorch)](https://pypi.org/project/unitorch/)
[![PyPI Version](https://badge.fury.io/py/unitorch.svg)](https://badge.fury.io/py/unitorch)
[![PyPI Downloads](https://pepy.tech/badge/unitorch)](https://pepy.tech/project/unitorch)
[![Github Downloads](https://img.shields.io/github/downloads/fuliucansheng/unitorch/total?color=blue&label=downloads&logo=github&logoColor=lightgrey)](https://img.shields.io/github/downloads/fuliucansheng/unitorch/total?color=blue&label=Downloads&logo=github&logoColor=lightgrey)

[![License](https://img.shields.io/github/license/fuliucansheng/unitorch?color=dfd)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/fuliucansheng/unitorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)

</div>

# Introduction
 
🔥 unitorch is a library that simplifies and accelerates the development of unified models for natural language understanding, natural language generation, computer vision, click-through rate prediction, multimodal learning and reinforcement learning. It is built on top of PyTorch and integrates seamlessly with popular frameworks such as transformers, peft, diffusers, and fastseq. With unitorch, you can use a single command line tool or a one-line code ` import unitorch` import to leverage the state-of-the-art models and datasets without sacrificing performance or accuracy.

------------------------------------

# What's New Model

* **SDXL** released with the paper [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) by Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach.
* **LLaMA** released with the paper [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.
* **ControlNet** released with the paper [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang, Anyi Rao, Maneesh Agrawala.
* **BLOOM** released with the paper [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) by BigScience Workshop: Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow...
* **PEGASUS-X** released with the paper [Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347) by Jason Phang, Yao Zhao, Peter J. Liu.
* **BLIP** released with the paper [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
* **BEiT** released with the paper [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) by Hangbo Bao, Li Dong, Songhao Piao, Furu Wei.
* **Swin Transformer** released with the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
* **CLIP** released with the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
* **mT5** released with the paper [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) by Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
* **Vision Transformer (ViT)** released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
* **DeBERTa-V2** released with the paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
* **DeBERTa** released with the paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
* **MBart** released with the paper [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
* **PEGASUS** released with the paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777) by Jingqing Zhang, Yao Zhao, Mohammad Saleh, Peter J. Liu.
* **BART** released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
* **T5** released with the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
* **VisualBERT** released with the paper [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557) by Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.
* **RoBERTa** released together with the paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
* **BERT** released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

------------------------------------

# Features

* User-Friendly Python Package
* Faster & Streamlined Train/Inference
* Deepspeed Integration for Large-Scale Models
* CUDA Optimization
* Extensive STOA Model & Task Supports

# Installation

```bash
pip3 install unitorch
```

# Quick Examples

### Source Code
```python
import unitorch

# import bart model
from unitorch.models.bart import BartForGeneration
model = BartForGeneration("path/to/bart/config.json")

# use the configuration class
from unitorch.cli import CoreConfigureParser
config = CoreConfigureParser("path/to/config.ini")
```

### Multi-GPU Training
```bash
torchrun --no_python --nproc_per_node 4 \
	unitorch-train examples/configs/generation/bart.ini \
	--train_file path/to/train.tsv --dev_file path/to/dev.tsv
```

### Single-GPU Inference
```bash
unitorch-infer examples/configs/generation/bart.ini --test_file path/to/test.tsv
```

> **Find more details in the Tutorials section of the [documentation](https://fuliucansheng.github.io/unitorch).**


# License

Code released under MIT license.
