# f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization

This repository contains the implementation of f-GAN using Variational Divergence Minimization.

## Introduction

f-GAN is a generative model training framework that uses a divergence measure to train both the generator and the discriminator. This implementation provides a flexible setup for experimenting with different divergence measures.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/arturpescador/fgan-variational-divergence-minimization.git
cd fgan-variational-divergence-minimization
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Usage

```bash
python train.py --epochs 100 --lr 0.002 --batch_size 64 --divergence <choose_divergence> --version <run_name>
```

#### Available divergence options:

- gan
- pr
- total_variation
- forward_kl
- reverse_kl
- pearson
- hellinger
- jensen_shannon


### Generating Samples

```bash
python generate.py --batch_size 2048
```

### TensorBoard

Visualize training progress using TensorBoard:

```bash
tensorboard --logdir=logs --port=6006
```

Visit http://localhost:6006 in your browser.

## Project Structure

The project follows the structure below:

- model.py: Contains the implementation of the Generator.
- utils.py: Utility functions.
- train.py: Script for training the f-GAN model.
- generate.py: Script for generating samples using the trained model.
- requirements.txt: List of dependencies.
- checkpoints/: Folder used for storing the trained model checkpoints.

- report.pdf: Project report.
- presentation.pdf: Project presentation.

## Contributors

This project was developed by the following members as part of the Data Science Lab course at Université Paris Dauphine for the academic year 2023/2024:

- Artur Dandolini Pescador
- Caio Jordan Azevedo

## References

The main papers consulted during the development of this project are:

1. **f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization**
   - **Authors:** Sebastian Nowozin, Botond Cseke, Ryota Tomioka
   - **Year:** 2016
   - **URL:** [arXiv:1606.00709](https://doi.org/10.48550/arXiv.1606.00709)

2. **Wasserstein GAN**
   - **Authors:** Martin Arjovsky, Soumith Chintala, Léon Bottou
   - **Year:** 2017
   - **URL:** [arXiv:1701.07875](https://doi.org/10.48550/arXiv.1701.07875)

3. **Precision-Recall Divergence Optimization for Generative Modeling with GANs and Normalizing Flows**
   - **Authors:** Alexandre Verine, Benjamin Negrevergne, Muni Sreenivas Pydi, Yann Chevaleyre
   - **Year:** 2023
   - **Note:** NeurIPS 2023
   - **URL:** [arXiv:2305.18910](https://doi.org/10.48550/arXiv.2305.18910)

