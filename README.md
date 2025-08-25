# CVRLOLD: Contrastive Video Representation Learning on Limited Datasets

This repository contains a reference implementation of the core components of the **CVRLoLD** framework, as presented in the paper:

> Ghelmani, A., & Hammad, A. (2023). Self-supervised contrastive video representation learning for construction equipment activity recognition on limited dataset. *Automation in Construction*, 154, 105001.

The project demonstrates a self-supervised learning approach for video activity recognition, specifically tailored for scenarios with limited data availability.

## About The Method

The core of this method is a **contrastive learning** pipeline that trains a 3D CNN backbone without requiring extensive labeled data. It learns robust spatiotemporal video representations by comparing different augmented views of video clips.

A key contribution demonstrated in this code is a **data sampling augmentation** technique, where multiple augmented clip pairs are extracted from each video in every training epoch. This strategy is crucial for successfully training the model on smaller datasets, a common challenge in specialized domains like construction monitoring.

## Repository Content

This repository includes the primary components for the self-supervised pre-training stage of the CVRLOLD method.

**Please Note:**
* This is not the complete code from the original research project. It has been simplified to highlight the core method.
* The dataset used for training and evaluation in the paper is not included in this repository.

For a comprehensive understanding of the framework, evaluation protocols, and results, please refer to the [full paper](https://doi.org/10.1016/j.autcon.2023.105001).

## Citation

If you find this work or code useful for your research, please consider citing our paper:

```bibtex
@article{ghelmani2023self,
  title={Self-supervised contrastive video representation learning for construction equipment activity recognition on limited dataset},
  author={Ghelmani, Ali and Hammad, Amin},
  journal={Automation in Construction},
  volume={154},
  pages={105001},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.autcon.2023.105001}
}
