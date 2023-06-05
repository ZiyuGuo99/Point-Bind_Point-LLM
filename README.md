# Point-Bind: Align 3D Point Clouds with Multi-modalities

This project presents **Point-Bind**, a 3D multi-modality model that aligns **3D point clouds** with image, language, audio, and videos guided by [ImageBind](https://github.com/facebookresearch/ImageBind).

## Overview

Our Point-Bind exhibits four main characters:

<p align="center">                                                                                                                                          <img src="point-bind.png"/ width="70%"> <br>
</p>

- **Align 3D with ImageBind.** With a joint embedding space, 3D objects can be aligned with their corresponding 2D images, textual descriptions, and audios.
- **3D LLM via LLaMA-Adapter.** Referring to [Multi-modal LLaMA-Adapter](), we introduce an LLM (LLaMA-Adapter) following 3D instructions ***for the first time***.
- **3D Zero-shot Classify/Seg/Det.** Point-Bind achieves ***state-of-the-art*** performance for 3D zero-shot tasks, including classification, segmentation, and detection.
- **Embedding Arithmetic with 3D.** We observe that 3D features from Point-Bind can be added with other modalities to compose their semantics.
