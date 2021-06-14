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

