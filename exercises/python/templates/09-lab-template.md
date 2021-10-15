---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region pycharm={"name": "#%% md\n"} -->
# Lab 9

**Authors**: Emilio Dorigatti, Tobias Weber

## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

set_matplotlib_formats('png', 'pdf')
```

<!-- #region pycharm={"name": "#%% md\n"} -->

## Exercise 1
In this exercise, we first train a logistic regression classifier on a subset of the
MNIST dataset, containing only the digits zero and one. Then, we look for adversarial
examples that can fool this classifier.

### Training logistic regression
First of all, we download the dataset (using torchvision), discard the samples we do
not need, normalize the inputs, turn them into vectors, add a bias term, and encode the
labels to $\pm 1$:

<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
class LogRegDataset(Dataset):
    def __init__(self, train: bool = True):
        data_raw = MNIST(root='.data', train=train, download=True)
        # PyTorch naming is a bit confusing here as `train_labels`
        # returns the test labels if test set is chosen.
        labels_raw = data_raw.train_labels
        self.y = labels_raw[labels_raw <= 1]
        self.y = torch.where(self.y == 1, 1, -1)

        img_filtered = data_raw.data[labels_raw <= 1]
        self.x = img_filtered.view(len(self.y), -1) / 255.
        self.x = torch.cat([torch.ones(len(self.y), 1), self.x], dim=1)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def shape(self) -> int:
        return self[0][0].shape

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


train_dataset = LogRegDataset(train=True)
test_dataset = LogRegDataset(train=False)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We now implement and train logistic regression, with loss function

\begin{equation}
\mathcal{L}(y, f(\textbf{x}|\theta))=\log\left(1+\exp\left(-y\cdot \theta^T\textbf{x}\right)\right)
\end{equation}

We could use PyTorch auto differentiation to do all the magic, but let's keep it
manually for the exercise.

<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def train_log_reg(
        dataset: LogRegDataset, max_steps: int, lr: float, batch_size: int
) -> Tuple[Tensor, List[float]]:
    theta = Normal(0, 0.001).sample(dataset.shape)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    step_counter = 0
    losses = []
    while True:
        for x, y in data_loader:
            #!TAG HWBEGIN
            #!MSG TODO: Perform the forward pass, compute predictions and loss
            linear = x @ theta
            loss = torch.sum(torch.log(1. + torch.exp(-y * linear))) / batch_size
            #!TAG HWEND
            losses.append(float(loss))

            #!TAG HWBEGIN
            #!MSG TODO: Perform the backward pass, compute the gradient of theta
            es = torch.exp(-y * linear)
            grad = -y * es / (1. + es) / batch_size
            grad = torch.sum(x.T * grad, dim=1)
            #!TAG HWEND

            #!TAG HWBEGIN
            #!MSG TODO: Perform the backward pass, compute the gradient of theta
            theta -= lr * grad
            #!TAG HWEND

            step_counter += 1
            if step_counter >= max_steps:
                return theta, losses

theta, losses = train_log_reg(train_dataset, max_steps=200, lr=0.1, batch_size=128)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plt.scatter(range(len(losses)), losses)
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We then compute the accuracy on the test set:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
test_predictions = torch.where(test_dataset.x @ theta > 0, 1, -1)
print(sum(test_predictions == test_dataset.y) / len(test_dataset))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Randomly searching adversarial examples
Now that the classifier is trained, we can construct adversarial examples.
The first strategy we try is to randomly generate vectors of a given length and check
whether they result in a different classification.

First, let's select an example that is classified correctly:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
x_sample = None
y_sample = None
for x, y in zip(train_dataset.x, train_dataset.y):
    prediction = torch.where(x @ theta > 0, 1, -1)
    if prediction == y:
        x_sample = x
        y_sample = y

plt.imshow(x_sample[1:].view(28, 28), cmap='gray')
plt.axis('off')
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Next, we write a function that perturbs this sample with a vector whose elements are all
$\pm\epsilon$, and checks whether the class is changed:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def perturb_and_check(x: Tensor, eps: float, y_true: int) -> bool:
    """Perturb sample and return a boolean that indicates the change of the class."""
    delta =(
        #!TAG HWBEGIN
        #!MSG TODO: Create a random vector with elements +/- eps
        eps * torch.sign(torch.randn(len(x)))
        #!TAG HWEND
    )

    prediction = (
        #!TAG HWBEGIN
        #!MSG TODO: Perturb the sample and predict its class
        torch.where((x + delta) @ theta > 0, 1, -1)
        #!TAG HWEND
    )

    return bool(y_true != prediction)

```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now, we try different values for `eps`. For each of them, we generate one thousand
different perturbations, and compute the proportion that result in a change of class:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

num_trials = 1000
eps_range = [1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2]

proportion_changed = []
for eps in eps_range:
    results = []
    for _ in range(num_trials):
        results.append(perturb_and_check(x_sample, eps, y_sample))
    proportion_changed.append(sum(results) / num_trials)

print(proportion_changed)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We can also plot this:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plt.scatter(eps_range, proportion_changed)
plt.xlabel('Half-width of box')
plt.ylabel('Proportion of mis-classified examples')
plt.xscale('log', base=10)
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Clearly, there are no points with a different classification up to a certain distance
from the original point. The specific range depends, obviously, on the point in question.
Further than that, and the proportion of points with a different class grows larger and
larger, until it stabilizes to a value smaller than one. Such range can be thought of
as the distance of the point to the decision boundary. If you now imagine a hyper-cube
centered on this point, the proportion of mis-classified examples is (an estimation of)
the amount of surface of this cube that is on the other side of the separating hyper-plane.
This intuition is valid regardless of the specific classification model employed, with
the only difference that the decision boundary can be arbitrarily more complex than a
hyper-plane.

### Creating an adversarial example via gradient ascent
Now, we can try to look for adversarial examples more intelligently.
Specifically, we want to find a vector $\delta^*$ such that

\begin{equation}
\delta^*=\text{argmax}_\delta \left[\mathcal{L}(y, f(\textbf{x}))-\mathcal{L}(y, f(\textbf{x}+\delta))\right]
\end{equation}

and $\delta\in\mathcal{B}^\infty_\epsilon$, with $f(x)$ being our trained logistic
regression classifier. The function above can be maximized with gradient ascent,
by slightly modifying the training procedure we employed earlier. In particular,
note that we can use values for epsilon for which a random perturbation has practically
no chance to result in a change of class.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def make_adversarial(
        theta: Tensor,
        sample: Tensor,
        y_true: int,
        eps: float,
        max_steps: int,
        lr
) -> Tuple[Tensor, List[float]]:
    delta = torch.randn(len(sample))
    y_wanted = -1 * y_true
    losses = []

    for _ in range(max_steps):

        #!TAG HWBEGIN
        #!MSG TODO: Perturb the sample with delta and compute the linear part of the model.
        linear = (x + delta) @ theta
        #!TAG HWEND


        loss_true = torch.sum(torch.log(1. + torch.exp(-y_true * linear)))
        loss_wanted = torch.sum(torch.log(1. + torch.exp(-y_wanted * linear)))

        #!TAG HWBEGIN
        #!MSG TODO: Compute the gradient of delta with respect to loss_true.
        es_true = torch.exp(-y_true * linear)
        grad_true = -y_true * theta * es_true / (1 + es_true)
        #!TAG HWEND

        #!TAG HWBEGIN
        #!MSG TODO: Compute the gradient of delta with respect to loss_true.
        es_wanted = torch.exp(-y_wanted * linear)
        grad_wanted = -y_wanted * theta * es_wanted / (1 + es_wanted)
        #!TAG HWEND

        #!TAG HWBEGIN
        #!MSG TODO: Update delta with one step of gradient ascent.
        #!MSG TODO: Normalize delta to have elements +/- eps.
        delta += lr * (grad_true - grad_wanted)
        delta = torch.clamp(delta, min=-eps, max=eps)
        #!TAG HWEND


        losses.append(loss_true - loss_wanted)
    return delta, losses


delta, losses = make_adversarial(
  theta=theta,
  sample=x_sample,
  y_true=y_sample,
  eps = 0.2,
  max_steps = 100,
  lr = 0.1
)

plt.scatter(range(len(losses)), losses)
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The label changes when the new loss becomes positive.
We can now check that the class was indeed changed:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
print('Original prediction:\t{:2}'.format(1 if x_sample @ theta > 0 else -1))
print('Modified prediction:\t{:2}'.format(1 if (x_sample + delta) @ theta > 0 else -1))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We can also visualize the resulting perturbation vector:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plt.imshow(delta[1:].view(28, 28), cmap='gray')
plt.axis('off')
plt.show()
```

and the perturbed sample:

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plt.imshow((x_sample + delta)[1:].view(28, 28), cmap='gray')
plt.axis('off')
plt.show()
```

<!-- #region -->
Can you interpret these images?

## Exercise 2
Suppose that the adversarial examples for a logistic regression classifier are generated
in $\mathcal{B}^2_\epsilon(\textbf{x})$ instead of $\mathcal{B}^\infty_\epsilon(\textbf{x})$.
Show that the adversarial risk becomes

\begin{equation}
\mathcal{L}(y,f(\textbf{x}|\theta))=\Psi\left(y(\theta^T\textbf{x})-\epsilon||\theta||_2\right)
\end{equation}


#!TAG HWBEGIN

### Solution
We can follow the same reasoning in the slides, i.e.

\begin{equation}
\delta^*=\max_{\delta\in\Delta(\textbf{0})}\mathcal{L}(y, f(\textbf{x} + \delta|\theta))=\ldots=\min_{\delta\in\Delta(\textbf{0})} y(\theta^T\delta)
\end{equation}

Now, however

\begin{equation}
\Delta(\textbf{0})=\mathcal{B}^2_\epsilon(\textbf{0})=\{ \delta :\delta\in\mathbb{R}^n \land ||\delta||_2 \leq \epsilon  \}
\end{equation}


When $y=1$, we have to find a vector $\delta$ of length $\epsilon$ that minimizes the
dot-product of $\theta$; such $\delta$ is headed in the opposite direction of $\theta$.
When $y=-1$, the dot-product has to be maximized, i.e. $\delta$ points in the same
direction of $\theta$. This means that

\begin{equation}
\delta^*=-y\epsilon\frac{\theta}{||\theta||_2}=\begin{cases}
-\epsilon\frac{\theta}{||\theta||_2} & y=1 \\
\epsilon\frac{\theta}{||\theta||_2} & y=-1 \\
\end{cases}
\end{equation}

Replacing this in the adversarial risk yields

\begin{equation}
\mathcal{L}(y,f(\textbf{x}|\theta))=\Psi\left(y(\theta^T\textbf{x})-y\theta^T\delta_*\right)=\Psi\left(y(\theta^T\textbf{x})-\epsilon\frac{\theta^T\theta}{||\theta||_2}\right)=\Psi\left(y(\theta^T\textbf{x})-\epsilon||\theta||_2\right)
\end{equation}


#!TAG HWEND
<!-- #endregion -->
