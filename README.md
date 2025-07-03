# Neural Network from Scratch

This project implements a **fully-connected feedforward neural network** from scratch using only **NumPy** — no TensorFlow, PyTorch, or other ML libraries. The model is trained and evaluated on the **MNIST** dataset for handwritten digit recognition.

---

## Features

- Custom implementation of:
  - Xavier (Glorot) weight initialization
  - ReLU and Softmax activations
  - Categorical Cross-Entropy loss
  - Forward and backward propagation
  - Gradient descent optimizer
- Vectorized operations for efficiency
- Modular, clean, beginner-friendly code
- Evaluates accuracy on both train and test sets

---

## Dataset: MNIST

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains **70,000** grayscale images of handwritten digits from 0 to 9. Each image is 28×28 pixels.

We load it using `keras.datasets.mnist`:

```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### Preprocessing Steps:

- Normalize pixel values to range `[0, 1]`
- Flatten images from (28, 28) to 784-dimensional vectors
- Transpose to shape `(features, samples)` for vectorized math
- One-hot encode labels

---

## Neural Network Architecture

| Layer         | Type           | Size         |
|---------------|----------------|--------------|
| Input         | Dense          | 784          |
| Hidden Layer  | Dense + ReLU   | customizable |
| Output        | Dense + Softmax| 10 classes   |

You can modify the number of hidden layers, neurons per layer, learning rate, and epochs easily in the notebook.

---

## Training Procedure

1. Initialize weights and biases (Xavier init)
2. Perform forward propagation
3. Compute loss using cross-entropy:
4. Perform backpropagation to compute gradients
5. Update weights using gradient descent:
6. Repeat over multiple epochs

---

## Results 
Achieves **~89–91% test accuracy** on MNIST

