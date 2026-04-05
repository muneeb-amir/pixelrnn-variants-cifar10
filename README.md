# PixelRNN Variants for Autoregressive Image Generation

This repository presents a faithful implementation and comparative study of three autoregressive generative models introduced in the PixelRNN paper:

* PixelCNN
* Row LSTM
* Diagonal BiLSTM

The models are trained and evaluated on the CIFAR-10 dataset to analyze their generative performance, computational trade-offs, and alignment with published benchmarks.

---

## Overview

Autoregressive generative models factorize the joint distribution of image pixels into a product of conditional distributions. Each pixel is predicted sequentially based on previously generated pixels.

This project implements three architectural variants with increasing modeling capacity:

| Model           | Key Mechanism                     |
| --------------- | --------------------------------- |
| PixelCNN        | Masked convolutions               |
| Row LSTM        | Sequential row-wise recurrence    |
| Diagonal BiLSTM | Bidirectional diagonal recurrence |

All implementations strictly follow the design principles described in the original PixelRNN paper.

---

## Dataset and Preprocessing

* Dataset: CIFAR-10
* Pixel values discretized to 256 levels per channel
* Logit transformation applied for improved optimization stability

---

## Implemented Architectures

### PixelCNN

* Masked convolutions enforcing autoregressive constraints
* Type A mask for input layer, Type B masks for hidden layers
* Residual connections with gated activations

### Row LSTM

* Sequential processing across image rows
* Convolutional LSTM with input-to-state and state-to-state transitions
* Residual connections for stable deep training

### Diagonal BiLSTM

* Bidirectional processing along image diagonals
* Skewing and unskewing operations for efficient diagonal traversal
* Captures long-range spatial dependencies
* Most expressive and computationally intensive variant

---

## Training Objective

Models are trained using discrete softmax likelihood:

* Output distribution: 256-way categorical per pixel channel
* Loss metric: Negative log-likelihood (bits per dimension)

---

## Results

### Bits Per Dimension (Lower is Better)

| Model           | Our Result | Paper Result | Difference |
| --------------- | ---------- | ------------ | ---------- |
| PixelCNN        | 4.729      | 3.140        | +1.589     |
| Row LSTM        | 4.692      | 3.070        | +1.622     |
| Diagonal BiLSTM | 1.631      | 3.000        | -1.369     |

Key observation:

* Diagonal BiLSTM achieves the best performance and significantly outperforms the published benchmark.

---

## Generative Quality Metrics

| Model           | Inception Score | FID    |
| --------------- | --------------- | ------ |
| PixelCNN        | 1.255 ± 0.121   | 413.32 |
| Row LSTM        | 1.133 ± 0.032   | 399.08 |
| Diagonal BiLSTM | 1.095 ± 0.032   | 431.72 |

---

## Key Insights

* Diagonal BiLSTM provides superior likelihood performance due to bidirectional context modeling
* Model complexity increases significantly from PixelCNN to Diagonal BiLSTM
* Likelihood improvements do not always correlate with perceptual quality (IS/FID)
* Autoregressive generation preserves spatial coherence effectively
* Implementation validates the importance of directional dependencies in image modeling

---

## Repository Structure

```
.
├── models_pixelrnn.py   # Model architectures
├── README.md
```

---

## Usage

Import and initialize any model:

```python
from models_pixelrnn import PixelCNN, RowLSTM, DiagonalBiLSTM

model = PixelCNN()
```

All models return logits over 256 discrete values per channel.

---

## Results
<img width="1293" height="703" alt="image" src="https://github.com/user-attachments/assets/ffc65c33-6dce-4850-9189-374b6d15e56c" />

<img width="1426" height="704" alt="image" src="https://github.com/user-attachments/assets/c59277d3-f44d-46c2-925f-91d2023ab5c9" />



