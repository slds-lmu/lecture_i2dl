---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region pycharm={"name": "#%% md\n"} -->
# Lab 2

**Lecture**: Deep Learning (Prof. Dr. David Rügamer, Emanuel Sommer)

## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
import torch
from torch import Tensor

import matplotlib.pyplot as plt
from  matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1
Consider a neural network with one hidden layer with threshold activation function
(i.e., $\tau(x)=\textbf{1}[x>0]$) and one output neuron with no activation function.
Prove that such a neural network cannot _exactly_ separate the cyan ($y=1$) and white
($y=0$) areas in the figure below.
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
![A region that cannot be classified correctly by a neural network
with a single hidden layer.](../utils/02-lab-f1.png)
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
**Hint**: start by assuming that such a neural network exists and has two neurons in the
 hidden layer with a generic activation function $\tau(\cdot)$. Consider four points, one for each of the four regions,
 $\textbf{x}_1,\ldots,\textbf{x}_4$ with $y_2=y_4=0$ and $y_1=y_3=1$.
 Compute the difference between the predictions of points with different labels,
 $g(\textbf{x}_1)-g(\textbf{x}_2)$ and $g(\textbf{x}_4)-g(\textbf{x}_3)$.
 You should reach a contradiction, meaning that such a neural network does not exists.

#!TAG HWBEGIN
### Solution
Assume that such a neural network exists. Then, it computes a function $g$ of the form:

\begin{equation*}
    g(\textbf{x})=\sum_{i=1}^m w_i\cdot \tau(\textbf{x}^T\textbf{v}_i+c_i)+b
\end{equation*}

Now consider four points as in the figure below, so that $g(\textbf{x}_1)=g(\textbf{x}_3)=1$
and $g(\textbf{x}_2)=g(\textbf{x}_4)=0$. Moreover, assume that the first hidden neuron
models $L_1$, so that it has a positive activation for $\textbf{x}_2$ and
$\textbf{x}_3$, and zero for $\textbf{x}_1$ and $\textbf{x}_4$.
<!-- #endregion -->

![Four points in the four regions.](../utils/02-lab-f2.png)


The difference between the predictions for $\textbf{x}_1$ and $\textbf{x}_2$ is

\begin{align*}
g(\textbf{x}_1)-g(\textbf{x}_2)
&= \sum_{i=1}^m w_i\cdot \tau(\textbf{x}_1^T\textbf{v}_i+c_i)+b-\sum_{i=1}^m w_i\cdot
\tau(\textbf{x}_2^T\textbf{v}_i+c_i)-b \\
&= w_1\cdot \tau(\textbf{x}_1^T\textbf{v}_1+c_1)-w_1\cdot \tau(\textbf{x}_2^T
\textbf{v}_1+c_1) \\
& =w_1\cdot 0-w_1\cdot 1 \\
&= -w_1 = 1
\end{align*}

The second step follows because when going from $\textbf{x}_1$ to $\textbf{x}_2$ we only
 cross $L_1$, and we stay on the same side of every other line. The last step follows
 because we know that $g(\textbf{x}_1)=1$ and that $g(\textbf{x}_2)=0$. This allows us
 to conclude that $w_1=-1$.

The same reasoning applies to $\textbf{x}_4$ and $\textbf{x}_3$, too:

\begin{align*}
g(\textbf{x}_4)-g(\textbf{x}_3)
&=w_1\cdot \tau(\textbf{x}_4^T\textbf{v}_1+c_1)-w_1\cdot
\tau(\textbf{x}_3^T\textbf{v}_1+c_1)) \\
&=-w_1 =-1
\end{align*}

Which implies that $w_1=1$. This contradicts what we found previously,
hence no such neural network exists.

Note: this proof is presented in

Blum, Edward K., and Leong Kwan Li. 1991.
“Approximation Theory and Feedforward Networks.”
 *Neural Networks* (4): 511–15. https://doi.org/10.1016/0893-6080(91)90047-9.7

#!TAG HWEND

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 2

In this exercise, we are going to create a neural network with two hidden layers and
threshold activation that can correctly separate the cyan region from the rest in the
plot below. We will do this by composing linear classifiers just like we did in the
previous lab.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
dots = torch.tensor([[-1, 1], [-1, -1], [0, 0], [1, 1], [1, -1]])
plt.figure()
plt.scatter(dots[:, 0], dots[:, 1])
t1 = plt.Polygon([[-1, 1], [-1, -1], [0, 0]], color='#b3cde3')
plt.gca().add_patch(t1)

t2 = plt.Polygon(dots[2:5,:], color='#b3cde3')
plt.gca().add_patch(t2)
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let us create a dataset containing points on a grid to test the network:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Build a grid of equally-spaced points, plus a column for the bias
grid_range = torch.linspace(-2, 2, 50)
grid_x, grid_y = torch.meshgrid(grid_range, grid_range, indexing='ij')
data = torch.stack([torch.ones(50**2), grid_x.flatten(), grid_y.flatten()]).T

print(data.shape)
print(data[:10])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### First hidden layer
The first hidden layer contains four neurons, each of which corresponds to a
line in the plot below.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
def plot_bowtie() -> None:
    plt.plot((1.25,-1.25),(1.25,-1.25),'k')
    plt.plot((1.25,-1.25),(-1.25,1.25),'k')
    plt.plot((-1,-1),(-1.5,1.5),'k')
    plt.plot((1,1),(-1.5,1.5),'k')

plot_bowtie()
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The following functions visualizes the decision boundary of a neuron with sigmoid
activation, $y=\sigma(a+bx_1+cx_2)$. You can use it to help you find the right
values for the weights.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def plot_decision_boundary(grid_x: Tensor, grid_y, pred: Tensor) -> None:
    """Plot the estimated decision boundary for a 2D grid with predictions."""
    plot_bowtie()
    plt.contourf(grid_x, grid_y, pred.view(grid_x.shape))
    plt.show()

def activation(x: Tensor) -> Tensor:
#!TAG HWBEGIN
#!MSG TODO apply the threshold activation to x.
  return torch.where(x > 0, 1., 0.)
#!TAG HWEND

def plot_decision_boundary_first_hidden(a: int, b: int, c: int) -> None:
    """Take 3 weights for one input neuron and plot resulting decision boundary."""
    neuron_output = (
#!TAG HWBEGIN
#!MSG TODO compute the output of the neuron with input weights given by a, b and c on
#!MSG each point of the grid.
    activation(data @ torch.tensor([a, b, c], dtype=torch.float))
#!TAG HWEND
    )
    plot_decision_boundary(grid_x, grid_y, neuron_output)

plot_decision_boundary_first_hidden(-1, -3, 1)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
For convenience, we group the parameters of the four neurons into a matrix with three
rows and five columns (one is for the bias), so that their output can be computed in a
single matrix multiplication. Each column contains the weights of a different neuron.
The first column contains a "fake" hidden neuron for the bias, whose value is always
one. Note that the first row of the weight matrix is connected to the bias of the
previous layer.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

weights_1 = torch.tensor([
    [1, 0, 0],   # bias neuron connected to the bias of the inputs
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the first neuron.
    [1, -1, 0],
    #!TAG HWEND
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the second neuron.
    [1, 1, 0],
    #!TAG HWEND
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the third neuron.
    [0, 1, -1],
    #!TAG HWEND
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the fourth neuron.
    [0, 1, 1],
    #!TAG HWEND
], dtype=torch.float).T
# Note: The weight matrix is transposed in the end. (5 x 3 --> 3 x 5)
# This gives more readability when entering your weights manually.

print(weights_1.shape)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let us plot the predictions of the four neurons:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_first_hidden(*weights_1[:, 1])
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_first_hidden(*weights_1[:, 2])
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_first_hidden(*weights_1[:, 3])
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_first_hidden(*weights_1[:, 4])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
And this is the first hidden layer of the network. Let us compute its predictions for
each point of the grid:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

hidden_1 = (
    #!TAG HWBEGIN
    #!MSG TODO compute the output of the first hidden layer using a matrix multiplication.
    activation(data @ weights_1)
    #!TAG HWEND
)

# Make sure that the number of rows is not changed,
print(len(hidden_1) == len(data))

# that there are five columns,
print(hidden_1.shape[1] == 5)

# and that the values are between zero and one
print(float(torch.min(hidden_1)), float(torch.max(hidden_1)))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Second hidden layer
The second hidden layer is composed of two neurons, each activating for inputs inside
one of the two triangles that make up our figure. These two neurons are connected to the
four neurons of the previous layer, thus each of them has five parameters.

Let us first create a new function to visualize the decision boundary of these
neurons in the second hidden layer.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def plot_decision_boundary_second_hidden (
        a: int, b: int, c: int, d: int, e: int) -> None:
    """Take 5 weights for one hidden neuron and plot resulting decision boundary."""
    neuron_output = (
    #!TAG HWBEGIN
    #!MSG TODO compute the output of the neuron.
    #!MSG (The input is the result from the first layer)
    activation(hidden_1 @ torch.tensor([a, b, c, d, e], dtype=torch.float))
    #!TAG HWEND
    )
    plot_decision_boundary(grid_x, grid_y, neuron_output)

plot_decision_boundary_second_hidden(-2, 3, -1, -3, 1)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now, as before, find the coefficients for the two neurons and put them into a matrix
with five rows and three columns. You can use the previous function to help you find
these weights.

**Hint:** You can think of these neurons as performing a logical AND operation on the
outputs of the neurons of the previous layer. All points inside each triangle must be
on the same side of three decision boundaries.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

weights_2 = torch.tensor([
    [1, 0, 0, 0, 0],   # bias neuron connected to the bias of the inputs
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the first neuron.
    [-2, 1, 0, 1, 1],
    #!TAG HWEND
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the second neuron.
    [-1, 1, 1, -1, -1],
    #!TAG HWEND
], dtype=torch.float).T
# Note: The weight matrix is transposed in the end. (3 x 5 --> 5 x 3)
# This gives more readability when entering your weights manually.

print(weights_2.shape)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_second_hidden(*weights_2[:, 1])
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_second_hidden(*weights_2[:, 2])
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

hidden_2 = (
    #!TAG HWBEGIN
    #!MSG TODO compute the output of the second hidden layer.
    activation(hidden_1 @ weights_2)
    #!TAG HWEND
)

# Make sure that the number of rows is not changed,
print(len(hidden_2) == len(data))

# that there are five columns,
print(hidden_2.shape[1] == 3)

# and that the values are between zero and one
print(float(torch.min(hidden_2)), float(torch.max(hidden_2)))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### Output layer
Finally, we can seek the parameters for the output neuron.
It should activate when an input is inside either one of the two triangles.

**Hint:** You can think of the output neuron as performing a logical OR operation on
the outputs of the second hidden layer.

Let us again modify the visualization function to show the decision of the network:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
def plot_decision_boundary_output (a: int, b: int, c: int) -> None:
    """Take 3 weights for one output neuron and plot resulting decision boundary."""
    neuron_output = (
    #!TAG HWBEGIN
    #!MSG TODO compute the output of the output neuron.
    #!MSG (The input is the result from the second layer)
    hidden_2 @ torch.tensor([a, b, c], dtype=torch.float)
    #!TAG HWEND
    )
    plot_decision_boundary(grid_x, grid_y, neuron_output)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now fill in the parameters:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

weights_3 = torch.tensor([
    #!TAG HWBEGIN
    #!MSG TODO find the coefficients for the output neuron.
    0, 1, 1
    #!TAG HWEND
], dtype=torch.float)

print(weights_3.shape)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The output of the neural network is:

<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary_output(*weights_3)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let us recap how the output is computed:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
hidden_1 = activation(data @ weights_1)
hidden_2 = activation(hidden_1 @ weights_2)
output = hidden_2 @ weights_3
```

<!-- #region pycharm={"name": "#%% md\n"} -->
This is, in essence, the *forward pass*.

If you did everything correctly, you should see below the bow tie image we wanted to reproduce:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plot_decision_boundary(grid_x, grid_y, output)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 3

**Note:** Focus on solving the second exercise first, as this exercise is of secondary
importance. Essentially, it shows that the same function can be computed by
many different neural networks.

Consider a neuron with incoming weights $\textbf{w}=w_1,\ldots,w_n$ bias $b$, and
activation $\tau(\cdot)$. This neuron is connected to the $i$-th neuron of the next
layer with the weight $v_i$, and the bias of the latter neuron is $c_i$. We want to
replace $\textbf{w}$, $v_i$, $b$ and $c_i$ with new parameters $\textbf{w}'$, $v'_i$,
$b'$ and $c'_i$ so that the output of the network is unchanged for all inputs.
At least one of the new parameters must be different, but some are allowed to equal
the old ones.

 1. Suppose that $\tau$ is the hyperbolic tangent. Show that the network computes the
same function if we let $\textbf{w}'=-\textbf{w}$, $v'_i=-v_i$, $b'=-b$ and $c'_i=c_i$.
 2. Now suppose that $\tau$ is the logistic sigmoid function. How should you set
$\textbf{w}'$, $v'_i$, $b'$ and $c'_i$? Hint: first, find the relationship between
$\sigma(x)$ and $\sigma(-x)$.
 3. Can you find other ways of modifying the parameters of a neural network without
altering its output? Equivalently, given a neural network computing a certain function,
how can you find a different network that computes the same function?

You do not have to provide a formal answer, but you can do so if you wish.

#!TAG HWBEGIN
### Solution
 1. The output of the neuron is $z'=\tanh(\textbf{w}'^T\textbf{x}+b')=\tanh(-\textbf{w}^T\textbf{x}-b)$.
Since $\tanh(x)=-\tanh(-x)$, we have $z'=-\tanh(\textbf{w}^T\textbf{x}+b)=-z$. Since $v'_i=-v_i$,
we have that $v'_iz'=(-v_i)\cdot(-z)=v_iz$. Therefore, no change to $c_i$ is necessary.
 2. We first prove that $\sigma(x)=1-\sigma(-x)$:

    \begin{align*}
    \sigma(x)-1+\sigma(-x)
    &=\frac{1}{1+e^{-x}}-1+\frac{1}{1+e^x} \\
    &=\frac{(1+e^x)-(1+e^{-x})(1+e^{x})+(1+e^{-x})}{(1+e^{-x})(1+e^{x})} \\
    &=\frac{1+e^x-1-e^x-e^{-x}-e^0+1+e^{-x}}{(1+e^{-x})(1+e^{x})} \\
    &=0
    \end{align*}

    We can follow the same idea as the previous question, and set
    $\textbf{w}'=-\textbf{w}$, $v_i'=-v_i$ and $b'=-b$. Then, we have
    $z'=\sigma(\textbf{w}'^T\textbf{x}+b')=\sigma(-\textbf{w}^T\textbf{x}-b')=1-\sigma(\textbf{w}^T\textbf{x}+b)=1-z$.
    Since $v'_i=-v_i$, the contribution of this neuron to the neurons of the next
    layer is $v'_iz'=-v_i(1-z)=-v_i+v_iz$, therefore we can set $c_i'=c_i+v_i$.
#!TAG HWEND
<!-- #endregion -->
