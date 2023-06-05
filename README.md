# Point-Bind: Align 3D Point Clouds with Multi-modalities

This project presents **Point-Bind** 🔥, a 3D multi-modality model that aligns **3D point clouds** with image, language, audio, and videos guided by [ImageBind](https://github.com/facebookresearch/ImageBind). The 3D encoder of Point-Bind is based on [I2P-MAE](https://github.com/ZrrSkywalker/I2P-MAE).

## Overview

Our Point-Bind exhibits four main characters:

<p align="center">                                                                                                                                          <img src="point-bind.png"/ width="70%"> <br>
</p>

- $\color{darkorange}{Align\ 3D\ with\ ImageBind\ .}$ With a joint embedding space, 3D objects can be aligned with their corresponding 2D images, textual descriptions, and audios.
- $\color{darkorange}{3D\ LLM\ via\ LLaMA-Adapter\ .}$ Referring to [Multi-modal LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main/imagebind_LLM), we introduce an LLM (LLaMA-Adapter) following 3D instructions ***for the first time***.
- $\color{darkorange}{3D\ Zero-shot\ Classify/Seg/Det\ .}$ Point-Bind achieves ***state-of-the-art*** performance for 3D zero-shot tasks, including classification, segmentation, and detection.
- $\color{darkorange}{Embedding\ Arithmetic\ with\ 3D\ .}$ We observe that 3D features from Point-Bind can be added with other modalities to compose their semantics.

## News
* The 3D instruction-following LLM via Multi-modal LLaMA-Adapter has been released, please referring to [ImageBind-LLM](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main/imagebind_LLM) 📌.
* The 3D zero-shot classification code of Point-Bind has been released 📌.


## Getting Started
Please refer to [Install.md](https://github.com/ZrrSkywalker/Point-Bind/blob/main/Install.md) for preparing environments and pre-trained checkpoints.

### 3D with Multi-modalities

### 3D LLM via LLaMA-Adapter

### 3D Zero-shot Tasks

Zero-shot classification accuracy comparison on ModelNet40 dataset:
|  Model | Encoder | Acc. (%)|
| :-----: | :-----: |:-----:|
|  [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP) | 2D CLIP |20.2|
|  [ULIP](https://github.com/salesforce/ULIP) | Point-BERT |60.4|
|  [PointCLIP V2](https://github.com/yangyangyang127/PointCLIP_V2) | 2D CLIP |64.2|
|  [ULIP 2](https://github.com/salesforce/ULIP) | Point-BERT |66.4|
|  Point-Bind | [Point-BERT](https://github.com/lulutang0608/Point-BERT) |76.3|
|  Point-Bind | [I2P-MAE](https://github.com/ZrrSkywalker/I2P-MAE) |**80.0**|


## Contributors
Renrui Zhang, Ziyu Guo, Xiangyang Zhu, Peng Gao

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn and zyguo@cse.cuhk.edu.hk.
