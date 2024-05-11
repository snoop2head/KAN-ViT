# KAN-ViT

## Introduction

This project aims to replace feedforward module of vision transformer (ViT) with Chebyshev Kolmogorov–Arnold Networks (KAN). 
- ViT and training framework is based on [affjljoo3581/deit3-jax](https://github.com/affjljoo3581/deit3-jax) which is great re-implementation of [DeiT](https://arxiv.org/abs/2012.12877).
- Chebyshev KAN implementation is based on [CG80499/KAN-GPT-2](https://github.com/CG80499/KAN-GPT-2), a project where GPT-2 style models are trained with KANs instead of MLPs.

### DeiT Reproduction
[affjljoo3581/deit3-jax](https://github.com/affjljoo3581/deit3-jax) project has trained ViTs using DeiT recipes with following results:

| Name | Data | Resolution | Epochs | Time | Reimpl. | Original | Config | Wandb | Model |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| T/16 | in1k | 224 | 300 | 2h 40m | 73.1% | 72.2% | [config](config/deit/deit-t16-224-in1k-300ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/icdx9h5c) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit-t16-224-in1k-300ep-best.msgpack?download=true) |
| S/16 | in1k | 224 | 300 | 2h 43m | 79.68% | 79.8% | [config](config/deit/deit-s16-224-in1k-300ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/hvp0ab58) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit-s16-224-in1k-300ep-best.msgpack?download=true) |
| B/16 | in1k | 224 | 300 | 4h 40m | 81.46% | 81.8% | [config](config/deit/deit-b16-224-in1k-300ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/98gmcuko) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit-b16-224-in1k-300ep-best.msgpack?download=true) |

Based on the DeiT reproduction, KAN-ViT is trained with the same configurations and hyperparameters. However, the results are suboptimal, underperforming the original DeiT recipes. Currently investigating the reasons for the performance drop.

### Environment Setup
Please refer to [affjljoo3581/deit3-jax](https://github.com/affjljoo3581/deit3-jax), where detailed step-by-step instructions are provided to setup the environment.

### Training
To train KAN-ViT, run one of the following scripts:
```bash
bash config/deit/kan-deit-b16-224-in1k-300ep.sh
bash config/deit/kan-deit-s16-224-in1k-300ep.sh
bash config/deit/kan-deit-t16-224-in1k-300ep.sh
```

## Hyperparameters

### ViT Architecture
* `--use-kan`: Flag to replace feedforward layers with Chebyshev KAN layers.
* `--polynomial-degree`: Degree of Chebyshev polynomial in KAN layers.
* `--layers`: Number of layers.
* `--dim`: Number of hidden features.
* `--heads`: Number of attention heads.
* `--labels`: Number of classification labels.
* `--layerscale`: Flag to enable LayerScale.
* `--patch-size`: Patch size in ViT embedding layer.
* `--image-size`: Input image size.
* `--posemb`: Type of positional embeddings in ViT. Choose `learnable` for learnable parameters and `sincos2d` for sinusoidal encoding.
* `--pooling`: Type of pooling strategy. Choose `cls` for using `[CLS]` token and `gap` for global average pooling.
* `--dropout`: Dropout rate.
* `--droppath`: DropPath rate.
* `--grad-ckpt`: Flag to enable gradient checkpointing for reducing memory footprint.

### Image Augmentations
* `--random-crop`: Type of random cropping. Choose `none` for nothing, `rrc` for RandomResizedCrop, and `src` for SimpleResizedCrop proposed in DeiT-III.
* `--color-jitter`: Factor for color jitter augmentation.
* `--auto-augment`: Name of auto-augment policy used in Timm (e.g. `rand-m9-mstd0.5-inc1`).
* `--random-erasing`: Probability of random erasing augmentation.
* `--augment-repeats`: Number of augmentation repetitions.
* `--test-crop-ratio`: Center crop ratio for test preprocessing.
* `--mixup`: Factor (alpha) for Mixup augmentation. Disable by setting to 0.
* `--cutmix`: Factor (alpha) for CutMix augmentation. Disable by setting to 0.
* `--criterion`: Type of classification loss. Choose `ce` for softmax cross entropy and `bce` for sigmoid cross entropy.
* `--label-smoothing`: Factor for label smoothing.

### Optimization
* `--optimizer`: Type of optimizer. Choose `adamw` for AdamW and `lamb` for LAMB.
* `--learning-rate`: Peak learning rate.
* `--weight-decay`: Decoupled weight decay rate.
* `--adam-b1`: Adam beta1.
* `--adam-b2`: Adam beta2.
* `--adam-eps`: Adam epsilon.
* `--lr-decay`: Layerwise learning rate decay rate.
* `--clip-grad`: Maximum gradient norm.
* `--grad-accum`: Number of gradient accumulation steps.
* `--warmup-steps`: Number of learning rate warmup steps.
* `--training-steps`: Number of total training steps.
* `--log-interval`: Number of logging intervals.
* `--eval-interval`: Number of evaluation intervals.

### Random Seeds
* `--init-seed`: Random seed for weight initialization.
* `--mixup-seed`: Random seed for Mixup and CutMix augmentations.
* `--dropout-seed`: Random seed for Dropout regularization.
* `--shuffle-seed`: Random seed for dataset shuffling.
* `--pretrained-ckpt`: Pretrained model path to load from.
* `--label-mapping`: Label mapping file to reuse the pretrained classification head for transfer learning.

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.

# Acknowledgement
- Thanks to the [TPU Research Cloud](https://sites.research.google/trc/about/) program for providing resources. Model is trained on the TPU `v3-8`.
- Thanks to [affjljoo3581](https://github.com/affjljoo3581) for the great re-implementation of DeiT and DeiT III.
- Thanks to [CG80499](https://github.com/CG80499) who provided Chebysev KAN implementation in JAX/FLAX.