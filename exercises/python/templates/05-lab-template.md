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
# Lab 4

**Authors**: Emilio Dorigatti, Tobias Weber

Welcome to the fifth lab. We will first implement a simple scalar automatic
differentiation engine to compute partial derivatives for us,
then do a theoretical exercise about L2 regularization.

## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import Tensor

set_matplotlib_formats('png', 'pdf')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1

Modern deep learning frameworks compute gradients automatically,
so that you only need to define how to perform the forward pass in your code.
Under the hood, the framework constructs a computational graph based on the operations
you used. For example, consider the node:

\begin{equation}
4xy+e^{-y}
\label{eq:ex}
\end{equation}

It can be translated into a graph that looks like this:

![](../utils/05-lab-graph1.png)

<!-- #endregion -->
