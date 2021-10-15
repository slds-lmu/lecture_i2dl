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

# Lab 3

**Authors**: Emilio Dorigatti, Tobias Weber

## Imports

```python pycharm={"name": "#%%\n"}
from typing import Any

import torch
from torch.autograd import Function
from torch import Tensor

import matplotlib.pyplot as plt
from  matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Welcome to the third lab. The first exercise is an implementation of gradient descent
on a bivariate function. The second exercise is about computing derivatives of the
weights of a neural network, and the third exercise combines the previous two.

## Exercise 1
This exercise is about gradient descent. We will use the function
$f(x_1, x_2)=(x_1-6)^2+x_2^2-x_1x_2$ as a running example:

 1. Use pen and paper to do three iterations of gradient descent:
     - Find the gradient of $f$;
     - Start from the point $x_1=x_2=6$ and use a step size of $1/2$ for the first step,
    $1/3$ for the second step and $1/4$ for the third step;
     - What will happen if you keep going?
 2. Write a function that performs gradient descent:
     - For simplicity, we use a constant learning rate.
     - Can you find a way to prematurely stop the optimization when you are close to the
    optimum?
     -
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
# Note: Defining a custom autograd function is not a necessity for this small task,
# but it is a good place to showcase some capabilities of PyTorch.

class MyFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        # The "ctx" object serves to stash information for the backward pass
        ctx.save_for_backward(x)
        func_value = (
            #!TAG HWBEGIN
            #!MSG TODO compute the value of f at x.
            (x[0] - 6)**2 + x[1]**2 - x[0] * x[1]
            #!TAG HWEND
        )
        return func_value

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        # The "grad_output" parameter is the backpropagated gradient from subsequent
        # operations w.r.t. to the output of this function.
        x = ctx.saved_tensors[0]

        grad_x = torch.tensor([
            #!TAG HWBEGIN
            #!MSG TODO compute the gradient of f at x.
            2*x[0] - x[1] - 12,
            -x[0] + 2*x[1]
            #!TAG HWEND
        ])
        return grad_output * grad_x
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

func = MyFunction()
# The "required_grad" argument needs to be True.
# Otherwise no gradients will be computed.
x = torch.tensor([6., 6.], requires_grad=True)

# Custom functions are applied over the "apply" method.
y = func.apply(x)
print('Function output: {}'.format(y))

# Gradients for every operation in this chain are computed
# by calling the "backward" method on the output tensor.
y.backward()

# The x tensor now has a grad attribute with the gradients.
print('Gradients: {}'.format(x.grad))

# Note: No usage of auto differentiation was done in this example.
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Does it match what you computed?

In the next step we define a small gradient descent optimizer.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

class GradientDescentOptimizer:
    def __init__(self,
                 func: Function,
                 max_steps: int,
                 alpha: float):
        """
        Init an Optimizer for performing GD.

        :param func: Function to apply.
        :param max_steps: Maximum number of GD steps.
        :param alpha: Learning Rate.
        """
        self.func = func
        self.max_steps = max_steps
        self.alpha = alpha

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply GD on a tensor.

        :param x: Input tensor.
        """
        # Usually you would apply the gradients inplace on the input tensor,
        # but for the sake of the example we keep the input tensor consistent and
        # work on a copy.
        x_cp = x.detach().clone()
        x_cp.requires_grad = True

        #!TAG HWBEGIN
        #!MSG TODO use a for loop to do gradient descent.
        #!MSG HINT When applying gradients you will need an "torch.no_grad()" context
        #!MSG manager. To modify the content of the tensor you will need its ".data"
        #!MSG attribute. Don't forget to erase the gradients after each iteration or
        #!MSG or they will accumulate.
        # Dummy value for initial loop.
        y_old = torch.tensor([float('inf')])

        for i in range(self.max_steps):
            # Set gradients of x to None
            x_cp.grad = None

            # Compute function output
            y = func.apply(x_cp)

            # Compute gradients
            y.backward()

            # Apply gradients
            # We need "no_grad" as otherwise the autodiff engine will compute
            # gradients for this operation.
            with torch.no_grad():
                x_cp.data -= self.alpha * x_cp.grad

            # Check if we reached some point of convergence.
            if y_old - y < 1e-4:
                break
            y_old = y
        #!TAG HWEND
        return x_cp
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
x = torch.tensor([6., 6.], requires_grad=True)
gd_optimizer = GradientDescentOptimizer(func=MyFunction(), max_steps=10, alpha=0.1)
x_new = gd_optimizer(x)
print(x_new)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Play a bit with the starting point and learning rate to get a feel for its behavior.
How close can you get to the minimum?

#!TAG HWBEGIN

### Solution

The gradient of $f$ is:

\begin{equation}
    \nabla_{\textbf{x}}f(\textbf{x})=\left\vert\begin{matrix}
    \partial f/\partial x_1 \\
    \partial f/\partial x_2 \\
    \end{matrix}\right\vert=\left\vert\begin{matrix}
    2(x_1-6)-x_2 \\
    2x_2-x_1 \\
\end{matrix}\right\vert
\end{equation}


For $\textbf{x}=|6,6|^T$ we have $f(\textbf{x})=0$ and $\nabla_{\textbf{x}}f(\textbf{x})=|-6,6|^T$.

Let $\textbf{x}^{(t)}$ denote the point at the $t$-th iteration. Then:

| $t$ | $\textbf{x}^{(t)}$ | $f(\textbf{x}^{(t)})$ | $\nabla_{\textbf{x}}f(\textbf{x})$ |
| --- | --- | --- | --- |
| 1 | $\mid 6,6 \mid$ | $0$ | $\mid-6,6\mid$ |
| 2 | $\mid 9,3\mid$ | $-9$ | $\mid3,-3\mid$ |
| 3 | $\mid8,4\mid$ | $-12$ | $\mid0,0\mid$  |

<!-- #endregion -->

1. $ x^{(2)} = | 6,6|-(1/2) * |-6,6| = |9,3|$
2. $ x^{(3)} = | 9,3|-(1/3) * |3,-3|=|8,4|$
3. $ x^{(4)} = | 8,4|-(1/4) * |0,0|=|8,4|$

Where all vectors are intended to be vertical. As the gradient at the last point is
zero, nothing will change if we continue to apply this procedure.

#!TAG HWEND

## Exercise 2

This exercise is about computing gradients with the chain rule, with pen and paper.
We will work with a neural network with a single hidden layer with two neurons and an
output layer with one neuron.

<!-- #region pycharm={"name": "#%% md\n"} -->
![Neural network used in Exercise 2](../utils/03-lab-nn.png)
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} -->
The neurons in the hidden layer use the $\tanh$ activation, while the output neuron uses
the sigmoid. The loss used in binary classification is the _binary cross-entropy_:

$$
\mathcal{L}(y, f_{out})=-y\log f_{out}-(1-y)\log(1-f_{out})
$$

where $y\in\{0,1\}$ is the true label and $f_{out}\in(0,1)$ is the predicted probability that $y=1$.

 1. Compute $\partial\mathcal{L}(y, f_{out})/\partial f_{out}$
 2. Compute $\partial f_{out}/\partial f_{in}$
 3. Show that $\partial\sigma(x)/\partial x=\sigma(x)(1-\sigma(x))$
 4. Show that $\partial\tanh(x)/\partial x=1-\tanh(x)^2$ (Hint: $\tanh(x)=(e^x-e^{-x})(e^x+e^{-x})^{-1}$)
 5. Compute $\partial f_{in}/\partial c$
 6. Compute $\partial f_{in}/\partial u_1$
 7. Compute $\partial\mathcal{L}(y,  f_{out})/\partial c$
 8. Compute $\partial\mathcal{L}(y,  f_{out})/\partial u_1$
 9. Compute $\partial f_{in}/\partial z_{2,out}$
 10. Compute $\partial z_{2,out}/\partial z_{2,in}$
 11. Compute $\partial z_{2,in}/\partial b_2$
 12. Compute $\partial z_{2,in}/\partial w_{12}$
 13. Compute $\partial z_{2,in}/\partial x_1$
 14. Compute $\partial\mathcal{L}(y,  f_{out})/\partial b_2$
 15. Compute $\partial\mathcal{L}(y,  f_{out})/\partial w_{12}$
 16. Compute $\partial\mathcal{L}(y,  f_{out})/\partial x_1$

You will notice that there are lots of redundancies. We will see how to improve these
computations in the lecture and in the next lab. Luckily, modern deep learning software
 computes gradients automatically for you.


#!TAG HWBEGIN

### Solution

#### Question 1
\begin{align*}
\frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
&=\frac{\partial}{\partial f_{out}}\bigg(y\log f_{out}+(1-y)\log(1- f_{out})\bigg)\\
&=-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}
\end{align*}

#### Question 2
\begin{align*}
\frac{\partial f_{out}}{\partial f_{in}}
&=\frac{\partial}{\partial f_{in}} \frac{1}{1+e^{- f_{in}}} \\
&=-(1+e^{- f_{in}})^{-2}\cdot -e^{- f_{in}} \\
&=\frac{e^{- f_{in}}}{(1+e^{- f_{in}})^{2}}
\end{align*}

#### Question 3
\begin{align*}
\frac{\partial}{\partial x} \frac{1}{1+e^{-x}}
&=\frac{e^{-x}}{(1+e^{-x})^2} \\
&=\frac{1}{1+e^{-x}} \cdot \frac{(1+e^{-x})-1}{1+e^{-x}} \\
&=\frac{1}{1+e^{-x}} \cdot \left(1-\frac{1}{1+e^{-x}}\right) \\
&=\sigma(x)(1-\sigma(x))
\end{align*}

#### Question 4
\begin{align*}
\frac{\partial}{\partial x}\tanh(x)
&=\frac{\partial}{\partial x}\frac{e^x-e^{-x}}{e^x+e^{-x}} \\
&=\frac{(e^x+e^{-x})(e^x+e^{-x})-(e^x-e^{-x})(e^x-e^{-x})}{(e^x+e^{-x})^2} \\
&=1-\frac{(e^x-e^{-x})^2}{(e^x+e^{-x})^2} \\
\end{align*}

#### Question 5

\begin{align*}
\frac{\partial f_{in}}{c}
&=\frac{\partial}{c}\left(c+u_1\cdot z_{1,out}+u_2\cdot z_{2,out}\right) \\
&=1
\end{align*}

#### Question 6

\begin{align*}
\frac{\partial f_{in}}{u_1}
&=\frac{\partial}{u_1}\left(c+u_1\cdot z_{1,out}+u_2\cdot z_{2,out}\right) \\
&=z_{1,out}
\end{align*}

#### Question 7

\begin{align*}
\frac{\partial\mathcal{L}(y,  f_{out})}{\partial c}
&=\frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
\cdot\frac{\partial f_{out}}{\partial f_{in}}
\cdot\frac{\partial f_{in}}{c} \\
&=\left(-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}\right)
\cdot\sigma(f_{in})(1-\sigma(f_{in}))
\cdot 1
\end{align*}


#### Question 8

\begin{align*}
\frac{\partial\mathcal{L}(y,  f_{out})}{\partial u_1}
&=\frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
\cdot\frac{\partial f_{out}}{\partial f_{in}}
\cdot\frac{\partial f_{in}}{u_1} \\
&=\left(-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}\right)
\cdot\sigma(f_{in})(1-\sigma(f_{in}))
\cdot z_{1,out}
\end{align*}


#### Question 9

\begin{align*}
\frac{\partial f_{in}}{\partial z_{2,out}}
&=\frac{\partial}{\partial z_{2,out}}\left(c+u_1\cdot z_{1,out}+u_2\cdot z_{2,out}\right) \\
&=u_2
\end{align*}

#### Question 10

\begin{align*}
\frac{\partial z_{2,out}}{\partial z_{2,in}}
&=\frac{\partial}{\partial z_{2,in}}\sigma(z_{2,in})\\
&=1-\tanh(z_{2,in})^2
\end{align*}

#### Question 11

\begin{align*}
\frac{\partial z_{2,in}}{\partial b_2}
&=\frac{\partial}{\partial b_2}\left(b_2+w_{12}\cdot x_1+w_{22}\cdot x_2 \right) \\
&=1
\end{align*}


#### Question 12

\begin{align*}
\frac{\partial z_{2,in}}{\partial w_{12}}
&=\frac{\partial}{\partial b_2}\left(b_2+w_{12}\cdot x_1+w_{22}\cdot x_2 \right) \\
&=x_1
\end{align*}

#### Question 13

\begin{align*}
\frac{\partial z_{2,in}}{\partial x_1}
&=\frac{\partial}{\partial x_1}\left(b_2+w_{12}\cdot x_1+w_{22}\cdot x_2 \right) \\
&=w_{12}
\end{align*}

#### Question 14
\begin{align*}
\frac{\partial\mathcal{L}(y,  f_{out})}{\partial b_2}
&=\frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
\cdot\frac{\partial f_{out}}{\partial f_{in}}
\cdot\frac{\partial f_{in}}{\partial z_{2,out}}
\cdot\frac{\partial z_{2,out}}{\partial z_{2,in}}
\cdot\frac{\partial z_{2,in}}{\partial b_2} \\
&=\left(-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}\right)
\cdot\sigma(f_{in})(1-\sigma(f_{in}))
\cdot u_2
\cdot(1-\tanh(z_{2,in})^2)
\cdot 1
\end{align*}

#### Question 15
\begin{align*}
\frac{\partial\mathcal{L}(y,  f_{out})}{\partial w_{12}}
&=\frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
\cdot\frac{\partial f_{out}}{\partial f_{in}}
\cdot\frac{\partial f_{in}}{\partial z_{2,out}}
\cdot\frac{\partial z_{2,out}}{\partial z_{2,in}}
\cdot\frac{\partial z_{2,in}}{\partial w_{12}} \\
&=\left(-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}\right)
\cdot\sigma(f_{in})(1-\sigma(f_{in}))
\cdot u_2
\cdot(1-\tanh(z_{2,in})^2)
\cdot x_1
\end{align*}

#### Question 16
\begin{align*}
\frac{\partial\mathcal{L}(y,  f_{out})}{\partial x_1}
&=\frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
\cdot\frac{\partial f_{out}}{\partial f_{in}}
\cdot\frac{\partial f_{in}}{\partial z_{2,out}}
\cdot\frac{\partial z_{2,out}}{\partial z_{2,in}}
\cdot\frac{\partial z_{2,in}}{\partial x_{1}} \\
&\quad\quad + \frac{\partial\mathcal{L}(y,  f_{out})}{\partial f_{out}}
\cdot\frac{\partial f_{out}}{\partial f_{in}}
\cdot\frac{\partial f_{in}}{\partial z_{1,out}}
\cdot\frac{\partial z_{1,out}}{\partial z_{1,in}}
\cdot\frac{\partial z_{1,in}}{\partial x_{1}} \\
&=\left(-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}\right)
\cdot\sigma(f_{in})(1-\sigma(f_{in}))
\cdot u_2
\cdot(1-\tanh(z_{2,in})^2)
\cdot w_{12} \\
&\quad\quad+\left(-\frac{y}{ f_{out}}+\frac{1-y}{1- f_{out}}\right)
\cdot\sigma(f_{in})(1-\sigma(f_{in}))
\cdot u_1
\cdot(1-\tanh(z_{1,in})^2)
\cdot w_{11}
\end{align*}

#!TAG HWEND

## Exercise 3

Now that we know how to do gradient descent and how to compute the derivatives of the
weights of a simple network, we can try to do these steps together and train our first
neural network! We will use the small dataset with five points we studied in the first
lab.

First, let's define the dataset:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
x = torch.tensor([
    [0, 0],
    [1, 0],
    [0, -1],
    [-1, 0],
    [0, 1]
], dtype=torch.float)
y = torch.tensor([1, 0, 0, 0, 0])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Next, a function to compute the output of the network:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def sigmoid(x: Tensor) -> Tensor:
#!TAG HWBEGIN
#!MSG TODO compute the sigmoid on x and return.
    return 1 / (1 + torch.exp(-x))
#!TAG HWEND

def predict(x: Tensor, b1: float, b2: float,
            w11: float, w12: float, w21: float, w22: float,
            c: float, u1: float, u2:float) -> Tensor:
    #!TAG HWBEGIN
    #!MSG TODO compute and return the output of the network.
    z1 = torch.tanh(b1 + x[:, 0] * w11 + x[:, 1] * w21)
    z2 = torch.tanh(b2 + x[:, 0] * w12 + x[:, 1] * w22)
    return sigmoid(c + u1 * z1 + u2 * z2)
    #!TAG HWEND

# This should return the predictions for the five points in the datasets
# We can unpack the param vector for the positional params of the function so that we don't
# need to enter every single entry.
params = torch.randn(9)
predictions = predict(x, *params)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Since gradient descent is done on the loss function, we need a function to compute it:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def get_loss(target: Tensor, pred: Tensor) -> Tensor:
    #!TAG HWBEGIN
    #!MSG TODO return the average loss.
    return -torch.mean(target * torch.log(pred + 1e-15) +
                       (1 - target) * torch.log(1 - pred + 1e-15))
    #!TAG HWEND

loss = get_loss(y, predictions)
print(loss)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now, we need to compute the gradient of each parameter:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def get_gradients(x: Tensor, target: Tensor,
            b1: float, b2: float,
            w11: float, w12: float, w21: float, w22: float,
            c: float, u1: float, u2:float) -> Tensor:
    # First, we perform the forward pass.
    z1in = b1 + x[:, 0] * w11 + x[:, 1] * w21
    z1out = torch.tanh(z1in)

    z2in = b2 + x[:, 0] * w12 + x[:, 1] * w22
    z2out = torch.tanh(z2in)

    fin = c + u1 * z1out + u2 * z2out
    fout = sigmoid(fin)

    #!TAG HWBEGIN
    #!MSG TODO compute all the partial derivatives.

    # Now we start back-propagation through the loss and the output neuron.
    dL_dfout = -target / (fout + 1e-15) + (1 - target) / (1 - fout + 1e-15)
    dfout_dfin = sigmoid(fin) * (1 - sigmoid(fin))

    # Compute the gradients for the parameters of the output layer.
    dfin_dc = 1
    dfin_du1 = z1out
    dfin_du2 = z2out

    # Take the mean gradient across data points
    dL_dc = torch.mean(dL_dfout * dfout_dfin * dfin_dc)
    dL_du1 = torch.mean(dL_dfout * dfout_dfin * dfin_du1)
    dL_du2 = torch.mean(dL_dfout * dfout_dfin * dfin_du2)

    # Back-propagate through the neurons in the first hidden layer.
    dfin_dz1out = u1
    dfin_dz2out = u2

    dz1out_dz1in = 1. - torch.tanh(z1in)**2
    dz2out_dz2in = 1. - torch.tanh(z2in)**2

    # Compute the derivatives of the parameters of the hidden layer.
    dz1in_db1 = dz2in_db2 = 1
    dL_db1 = torch.mean(dL_dfout * dfout_dfin * dfin_dz1out * dz1out_dz1in * dz1in_db1)
    dL_db2 = torch.mean(dL_dfout * dfout_dfin * dfin_dz2out * dz2out_dz2in * dz2in_db2)

    dz1in_dw11 = dz2in_dw12 = x[:, 0]
    dL_dw11 = torch.mean(dL_dfout * dfout_dfin * dfin_dz1out * dz1out_dz1in * dz1in_dw11)
    dL_dw12 = torch.mean(dL_dfout * dfout_dfin * dfin_dz2out * dz2out_dz2in * dz2in_dw12)

    dz1in_dw21 = dz2in_dw22 = x[:, 1]
    dL_dw21 = torch.mean(dL_dfout * dfout_dfin * dfin_dz1out * dz1out_dz1in * dz1in_dw21)
    dL_dw22 = torch.mean(dL_dfout * dfout_dfin * dfin_dz2out * dz2out_dz2in * dz2in_dw22)

    #!TAG HWEND
    
    # Return the derivatives in the same order as the parameters vector
    return torch.stack([
        dL_db1, dL_db2, dL_dw11, dL_dw12, dL_dw21, dL_dw22, dL_dc, dL_du1, dL_du2  
    ])

print(get_gradients(x, y, *params))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Finite differences are a useful way to check that the gradients are computed correctly:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

# First, compute the analytical gradient of the parameters.
gradient = get_gradients(x, y, *params)
eps = 1e-9
for i in range(9):
    # Compute loss when subtracting eps to parameter i.
    neg_params = params.clone()
    neg_params[i] = neg_params[i] - eps
    neg_value = get_loss(y, predict(x, *neg_params))

    # Compute loss when adding eps to parameter i.
    pos_params = params.clone()
    pos_params[i] = pos_params[i] + eps
    pos_value = get_loss(y, predict(x, *pos_params))

    # Compute the "empirical" gradient of parameter i
    fdiff_gradient = torch.mean((pos_value - neg_value) / (2 * eps))

    # Error if difference is too large
    if torch.abs(gradient[i] - fdiff_gradient) < 1e-5:
        raise ValueError('Gradients are probably wrong!')

print("Gradients are correct!")
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We can finally train our network. Since the network is so small compared to the dataset,
 the training procedure is very sensitive to the way the weights are initialized and
 the step size used in gradient descent.

Try to play around with the learning rate and the random initialization of the weights
and find reliable values that make training successful in most cases.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

min_loss = 10
alpha = 1.
steps = 100
best_params = None

for i in range(10):
    params = torch.randn(9)

    # Do GD
    for _ in range(steps):
        gradients = get_gradients(x, y, *params)
        params -= alpha * gradients

    final_loss = get_loss(y, predict(x, *params))
    print('RUN {} \t LOSS {:.4f}'.format(i + 1, float(final_loss)))

    if final_loss < min_loss:
        best_params = params
        min_loss = final_loss
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We can use the function in the previous lab to visualize the decision boundary of
the best network:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def plot_decision_boundary(
        x: Tensor, y: Tensor, grid_x: Tensor, grid_y, pred: Tensor) -> None:
    """Plot the estimated decision boundary for a 2D grid with predictions."""
    plt.contourf(grid_x, grid_y, pred.view(grid_x.shape))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='jet')
    plt.show()
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

grid_range = torch.linspace(-2, 2, 50)
grid_x, grid_y = torch.meshgrid(grid_range, grid_range)
grid_data = torch.stack([grid_x.flatten(), grid_y.flatten()]).T
pred = predict(grid_data, *best_params)

plot_decision_boundary(x, y, grid_x, grid_y, pred)


```

<!-- #region pycharm={"name": "#%% md\n"} -->
Also try to visualize the decision boundary of network with random parameters:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
pred = predict(grid_data, *torch.randn(9))
plot_decision_boundary(x, y, grid_x, grid_y, pred)
```
