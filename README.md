# MLPMixer

## Introduction
MLP Mixers was published in May 2021 by Brain team with the impressive [paper](https://arxiv.org/abs/2105.01601v1).
Their work presented a simple, efficient and speed solution for computer vision and can be a new competitor against CNNs
and transformer in the future. The new architecture contains two type of layers: token mixing and channel mixing. This network
is actually lightweight with the computational complexity is linear comparable with the quadratic approximate in ViT and can work
very well without the position embedding which is a vital element in transformer-based model. In our work, I'll assess this architecture 
for classification task. 

## Architecture
![](images/architecture.png)

## Training 
```bash
$ python train.py --help

usage: train.py [-h] [--image_shape IMAGE_SHAPE] [--patch_shape PATCH_SHAPE]
                [--depth DEPTH] [--dim DIM] [--num_classes NUM_CLASSES]
                [--checkpoint CHECKPOINT] [--output_dir OUTPUT_DIR]
                [--batch_size BATCH_SIZE] [--lr LR] [--momentum MOMENTUM]
                [--decay DECAY] [--num_steps NUM_STEPS]
                [--warmup_step WARMUP_STEP] [--scheduler {cosine,linear}]

Training process

optional arguments:
  -h, --help            show this help message and exit
  --image_shape IMAGE_SHAPE
                        Image shape for training (default: (256, 256))
  --patch_shape PATCH_SHAPE
                        Patch shape for splitting (default: (16, 16))
  --depth DEPTH         Depth of model (default: 6)
  --dim DIM             Dim for each mlp mixer block (default: 512)
  --num_classes NUM_CLASSES
                        Number of classes for classification (default: 120)
  --checkpoint CHECKPOINT
                        Path to checkpoint (default: None)
  --output_dir OUTPUT_DIR
                        Path to save checkpoint (default: output/)
  --batch_size BATCH_SIZE
                        Number of images in each batch (default: 16)
  --lr LR               Learning rate (default: 1e-3)
  --momentum MOMENTUM   Momentum (default: 0,9)
  --decay DECAY         Weight decay (default: 0)
  --num_steps NUM_STEPS
                        Number of epochs (default: 30)
  --warmup_step WARMUP_STEP
                        Number of epoch for warmup learning rate (default: 10)
  --scheduler {cosine,linear}
                        Scheduler for warmup learning rate (default: cosine)


```

## Citations
```bibtex
@article{tolstikhin2021,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}
```
