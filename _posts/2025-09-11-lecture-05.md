---
title: "Lecture 05"
layout: distill
date: "2025-09-17"
description: Fitting Neurons with Gradient Descent
lecturers:
- name: Ben Lengerich
  url: https://lengerichlab.github.io/
authors:
- name: Anna Schellin
- name: Yuheng Mao
- name: Flora He
---
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
## Lecture Overview

1. Online, batch, and minibatch mode
2. Relation between percetron and linear regression
3. An iterative training algorithm for linear regression
4. Calculus Refresher I: Derivatives
5. Calculus Refresher II: Gradients
6. Understanding gradient descent
7. Training an adaptive linear neuron (Adaline) 

---

## 1. Online, batch, and minibatch mode
### Perceptron Recap

<p style="text-align:center;">
  <img src="{{ '/assets/img/notes/lecture-05/perceptron_recap.png' | relative_url }}" style="max-width:35%; max-width:35%;"/>
</p>

<p style="text-align:center;">
, where
</p>

$$
\sigma \!\Bigl( \sum_{i=1}^{m} x_i w_i + b \Bigr)
= \sigma \!\bigl( \mathbf{x}^{T} \mathbf{w} + b \bigr)
= \hat{y}
$$

$$
\sigma(z) =
\begin{cases}
0, & z \le 0 \\
1, & z > 0
\end{cases}
$$

$$
b = -\theta
$$

<hr style="border: 1px dashed #999;">

Let $ \mathcal{D} = (\langle \mathbf{x}^{[1]}, y^{[1]} \rangle, \mathbf{x}^{[2]}, y^{[2]} \rangle, \dots, \mathbf{x}^{[n]}, y^{[n]} \rangle) \in (\mathbb{R}^{m} \times \{0,1\})^{n} $.

1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $
2. For every training epoch:

    A. For every $ \langle x^{[i]}, y^{[i]} \rangle \in \mathcal{D} $:

    - (a) Compute output (prediction) $ \hat{y}^{[i]} := \sigma\( x^{[i]\top} \mathbf{w} + b \bigr) $  

    - (b) Calculate error $ \mathrm{err} := \bigl( y^{[i]} - \hat{y}^{[i]} \bigr) $  

    - (c) Update parameters $ \mathbf{w} := \mathbf{w} + \mathrm{err} \times x^{[i]}, \quad b := b + \mathrm{err} $  

### "On-line" mode (= SGD: Schocastic Gradient Descent)

1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $
2. For every training epoch:

    A. For **every** $ \langle x^{[i]}, y^{[i]} \rangle \in \mathcal{D} $:

    - (a) Compute output (prediction)
    - (b) Calculate error
    - (c) Update $ \mathbf{w}, \mathbf{b}$
  
- For step 2, we usually <u>shuffle</u> the dataset prior to each epoch to prevent cycles
- Applies to all common neuron models and (deep) neural network architectures

### "On-line" mode II (alternative)

1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $
2. For every training epoch:

    A. **Pick random** $ \langle x^{[i]}, y^{[i]} \rangle \in \mathcal{D} $:

    - (a) Compute output (prediction)
    - (b) Calculate error
    - (c) Update $ \mathbf{w}, \mathbf{b}$
  
- No shuffling required

### Batch mode
1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $
2. For every training epoch:

    A. Take **all** training examples from $\mathcal{D} $:

    - (a) Compute output (prediction)
    - (b) Calculate error

    B. Update $ \mathbf{w}, \mathbf{b}$
    
### Minibach mode (mix between on-line and batch)
1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $
2. For every training epoch:

    A. For every **minibatch** of size k, namely 
    
    $ (\langle x^{[i]}, y^{[i]} \rangle, \dots, \langle x^{[i+k]}, y^{[i+k]} \rangle) \in \mathcal{D}$

    - (a) Compute output (prediction)
    - (b) Calculate error
    - (c) Update $ \mathbf{w} := \mathbf{w} + \Delta \mathbf{w}, \quad \mathbf{b} := \mathbf{b} + \Delta \mathbf{b} $

- This is the **most common** mode in deep learning because:
  1. choosing a subset instead of one example at a time <u>takes advantage of vectorization</u> (faster iteration through epoch than on-line)
  2. having fewer updates than "on-line" makes updates <u>less noisy</u>. 
  3. makes more updates / epoch than "batch" and is thus <u>faster</u>
  
---

## 2. Relation between perceptron and linear regression
### Perceptron
  - activation function is the threshold function
  - output is a binary label $\hat{y} \in \({0,1\})$
  
### Linear Regression
  - activation function is the identity function $ \sigma(x) = x $
  - output is a real number $\hat{y} \in \mathbb{R}$
  - you can think of linear regression as a linear neuron
  
### (Least-Squares) Linear Regression

$$
\mathbf{w} = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}y
$$

, assuming the bias is included in $ \mathbf{w}$, and the design matrix has an additional vector of 1's

- Generally, this is the best approach for linear regression

---

## 3. An iterative training algorithm for linear regression

### (Least-Squares) Linear Regression
* A very naive way to fit a linear regression model (and any neural net) is to start with initializing the parameters to 0's or small random values 
* Then, for k rounds 

  1. Choose another random set of weights
  2. If the model performs **better**, **keep** those weights 
  3. If the model performs **worse**, **discard** the weights

* Guaranteed to find the optimal solution for very large k, but it would be terribly slow

### Better way
* analyze what effect a change of a parameter has on the predictive performance (loss) of a model
* then change the weight a little in the direction that improves the performance (minimizes the loss) the most
* do this in several (small) steps until the loss does not further decrease

### Update Rules ("on-line" mode)
#### Perceptron Learning Rule
1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $
2. For every training epoch:

    A. For every $ \langle x^{[i]}, y^{[i]} \rangle \in \mathcal{D} $:

    - (a) $ \hat{y}^{[i]} := \sigma(x^{[i]\top}\mathbf{w} + b) $
    - (b) $ err := (y^{[i]} - \hat{y}^{[i]}) $
    - (c) $ \mathbf{w} := \mathbf{w} + err \times x^{[i]}, \quad b := b + err $

#### Stochastic Gradient Descent (Vectorized)
1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $

2. For every training epoch:

    A. For every $ \langle x^{[i]}, y^{[i]} \rangle \in \mathcal{D} $:

      - (a) $ \hat{y}^{[i]} := \sigma(x^{[i]\top}\mathbf{w} + b) $

      - (b) $ \nabla_{\mathbf{w}}\mathcal{L} = (y^{[i]} - \hat{y}^{[i]})x^{[i]}, \quad
                \nabla_{b}\mathcal{L} = (y^{[i]} - \hat{y}^{[i]}) $

      - (c) $ \mathbf{w} := \mathbf{w} + \eta \cdot (-\nabla_{\mathbf{w}}\mathcal{L}), \quad
                b := b + \eta \cdot (-\nabla_{b}\mathcal{L}) $
      
      where $\eta$ = learning rate,  $(-\nabla_{\mathbf{w}}\mathcal{L})$ = negative gradient
  
#### Stochastic Gradient Descent (For understanding only)
1. Initialize $ \mathbf{w} := \mathbf{0} \in \mathbb{R}^{m}, \quad b := 0 $

2. For every training epoch:

    A. For every $ \langle x^{[i]}, y^{[i]} \rangle \in \mathcal{D} $:

      - (a) $ \hat{y}^{[i]} := \sigma(x^{[i]\top}\mathbf{w} + b) $
   
   **B. For weight** $ j \in \{1, \ldots, m\}$**:**

      - (b) $ \frac{\partial \mathcal{L}}{\partial w_{j}} =
                     (y^{[i]} - \hat{y}^{[i]})x_{j}^{[i]} $

      - (c) $ w_{j} := w_{j} + \eta \cdot
                     \Bigl(-\frac{\partial \mathcal{L}}{\partial w_{j}}\Bigr) $

    C. $ \frac{\partial \mathcal{L}}{\partial b} =
          (y^{[i]} - \hat{y}^{[i]}), 
          b := b + \eta \cdot \Bigl(-\frac{\partial \mathcal{L}}{\partial b}\Bigr) $
          
      where $\eta \cdot \Bigl(-\frac{\partial \mathcal{L}}{\partial b}\Bigr)$ coincidentally appears almost to be the same as the perceptron rule, except that the prediction is a **real number**, and we have a **learning rate**

### This learning rule is called <u>Gradient Descent</u>
