# Installation

## Requirements

- python version >= 3.8
- [torch](http://pytorch.org/) >= 1.13
- [torchvision](http://pytorch.org/)
- [torchaudio](http://pytorch.org/)
- fire
- configparser
- pandas <= 1.5.3
- scikit-learn >= 0.24.2
- [diffusers](https://github.com/huggingface/diffusers) >= 0.16.1
- [deepspeed](https://github.com/microsoft/deepspeed) >= 0.9.0
- [peft](https://github.com/huggingface/peft) >= 0.3.0
- [datasets](https://github.com/huggingface/datasets) >= 2.12.0
- [transformers](https://github.com/huggingface/transformers) >= 4.29.1


## Install Pypi

```bash
pip3 install unitorch
pip3 install unitorch[deepspeed]
```

## Install Source

```bash
pip3 install \
    "git+https://github.com/fuliucansheng/unitorch#egg=unitorch[deepspeed]"
```

## Install Extension

```bash
UNITORCH_EXTENSIONS=NGRAM pip3 install \
    "git+https://github.com/fuliucansheng/unitorch"
```