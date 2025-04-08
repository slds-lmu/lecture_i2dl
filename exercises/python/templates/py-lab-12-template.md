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
# Lab 11 - currently not in use

## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}

import matplotlib.pyplot as plt
import torch
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

set_matplotlib_formats('png', 'pdf')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1


In this lab we are going to train a generative adversarial network (GAN) on the MNIST
dataset using PyTorch. A useful set of tricks to train GANs can be found at
[https://github.com/soumith/ganhacks](https://github.com/soumith/ganhacks).

As usual, we start by loading and normalizing the MNIST dataset.
We also get our device.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
dataset = MNIST(root='.data', download=True, transform=ToTensor());
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Next, we create the generator network. This time, we are going to use an optimizer with
customized parameters. Specifically, we use the Adam optimizer with a learning rate of
0.0002, $\beta_1=0.5$ and $\beta_1=0.999$. This will give more weights to recent gradient updates.

The generator takes as input normally-distributed, 8-dimensional noise vectors and
should be composed of four fully-connected layer of sizes 256, 512, 1024 and 784, each
using a leaky ReLU activation with $\alpha=0.2$ except for the output layer which uses
$\tanh$.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC


generator = nn.Sequential(
    #!TAG HWBEGIN
    #!MSG Define the network as in the description above
    nn.Linear(in_features=8, out_features=256),
    nn.LeakyReLU(0.2),
    nn.Linear(in_features=256, out_features=512),
    nn.LeakyReLU(0.2),
    nn.Linear(in_features=512, out_features=1024),
    nn.LeakyReLU(0.2),
    nn.Linear(in_features=1024, out_features=784),
    nn.Tanh(),
    #!TAG HWEND
).to(device)

gen_optimizer = (
    #!TAG HWBEGIN
    #!MSG Define the optimizer as in the description above
    Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    #!TAG HWEND
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We now create the discriminator with a mirrored architecture, i.e. layers of size 1024,
512, 256 and 1. As before, use the leaky ReLU activation with $\alpha=0.2$, and add
dropout with $p=0.3$ between all layers.
As the discriminator will perform a binary classification task, use the sigmoid
activation in the output layer. Use the same kind of optimizer you created earlier.

<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC


discriminator = nn.Sequential(
    #!TAG HWBEGIN
    #!MSG Define the network as in the description above
    nn.Linear(in_features=784, out_features=1024),
    nn.LeakyReLU(0.2),
    nn.Dropout(),
    nn.Linear(in_features=1024, out_features=512),
    nn.LeakyReLU(0.2),
    nn.Dropout(),
    nn.Linear(in_features=512, out_features=256),
    nn.LeakyReLU(0.2),
    nn.Dropout(),
    nn.Linear(in_features=256, out_features=1),
    nn.Sigmoid(),
    #!TAG HWEND
).to(device)

dis_optimizer = (
    #!TAG HWBEGIN
    #!MSG Define the optimizer as in the description above
    Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    #!TAG HWEND
)

```

<!-- #region pycharm={"name": "#%% md\n"} -->
In the next step we will define and run our training loop, which will be a bit more extensive
and usual. In praxis, a lot of these complexities should be kept as abstractions and
individual classes. In this exercise we focus more on the actual steps to train a GAN.
The generator and discriminator updates are computed in separate steps.

We will train the GAN for around 8000 weight updates (aka 8 epochs with batch_size 64).
GANs can also overfit, but more training would not hurt in this case!
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
epochs = 8
batch_size = 64
latent_size = 8

# We will use the BCE loss in combination with constructed labels to train the GAN.
criterion = nn.BCELoss()
real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

# Tracking the losses during training
gen_losses = []
dis_losses = []

train_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

batch_updates = 0
for ep in range(1, epochs + 1):
    for x, _ in train_loader:
        # Push to device and flatten the images to vectors
        x = x.to(device).view(x.shape[0], -1)
        # Scale the vectors from [0, 1] -> [-1, 1]
        x = x * 2 - 1

        ####################################################
        # UPDATE DISCRIMINATOR
        ####################################################

        # The noise is the input for the generator
        noise = (
            #!TAG HWBEGIN
            #!MSG Create some normal distributed noise with shape (batch_size, latent_size).
            torch.randn((batch_size, latent_size), device=device)
            #!TAG HWEND
        )

        # We don't need generator gradients for this step
        with torch.no_grad():
            # We generate new fake data.
            fake_x = (
                #!TAG HWBEGIN
                #!MSG Generate some fake samples with the generator and noise
                generator(noise)
                #!TAG HWEND
            )

        # Compute the discriminator score for the fake data
        fake_score = (
            #!TAG HWBEGIN
            #!MSG Compute the discriminator score for the fake samples.
            discriminator(fake_x)
            #!TAG HWEND
        )

        # Compute the discriminator score for the real data
        real_score = (
            #!TAG HWBEGIN
            #!MSG Compute the discriminator score for the real samples.
            discriminator(x)
            #!TAG HWEND
        )

        # Compute the discriminator loss for the real data
        real_loss = (
            #!TAG HWBEGIN
            #!MSG Compute the discriminator loss for the real data (Hint: Use the criterion and real labels)
            criterion(real_score, real_labels)
            #!TAG HWEND
        )

        # Compute the discriminator loss for the real data
        fake_loss = (
            #!TAG HWBEGIN
            #!MSG Compute the discriminator loss for the fake data (Hint: Use the criterion and fake labels)
            criterion(fake_score, fake_labels)
            #!TAG HWEND
        )

        dis_loss = (real_loss + fake_loss) / 2

        dis_optimizer.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

        ####################################################
        # UPDATE GENERATOR
        ####################################################

        # The noise is the input for the generator
        noise = (
            #!TAG HWBEGIN
            #!MSG Create some normal distributed noise with shape (batch_size, latent_size).
            torch.randn((batch_size, latent_size), device=device)
            #!TAG HWEND
        )

        # We generate new fake data (and allow gradients now!)
        fake_x = (
                #!TAG HWBEGIN
                #!MSG Generate some fake samples with the generator and noise
                generator(noise)
                #!TAG HWEND
        )

        # Compute the discriminator score for the fake data
        fake_score = (
            #!TAG HWBEGIN
            #!MSG Compute the discriminator score for the fake samples.
            discriminator(fake_x)
            #!TAG HWEND
        )

        # Compute the generator loss
        gen_loss = (
            #!TAG HWBEGIN
            #!MSG Compute the generator GAN loss. Hint: Criterion + real labels!
            criterion(fake_score, real_labels)
            #!TAG HWEND
        )
        
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        batch_updates += 1
        gen_losses.append(float(gen_loss))
        dis_losses.append(float(dis_loss))

        if batch_updates % 100 == 0:
            print('BATCH UPDATE:\t{:5}\tDIS LOSS:\t{:2.3f}\tGEN LOSS:\t{:2.3f}'
                  .format(batch_updates, float(dis_loss), float(gen_loss)), end='\r')
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
plt.plot(gen_losses, label='Generator Loss')
plt.plot(dis_losses, label='Discriminator Loss')
plt.legend()
plt.show()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Training did not fully converge, as evidenced by the generative loss still decreasing
and the discriminative loss increasing at the end of training.

Finally, we can try to use the trained GAN to generate new samples by feeding random
noise to the generator network:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
noise = torch.randn(100, latent_size, device=device)
fake_samples = (generator(noise).view(100, 1, 28, 28).detach().cpu() + 1) / 2

image_grid = make_grid(fake_samples, nrow=20)
plt.imshow(image_grid.permute(1, 2, 0))
plt.axis('off')
plt.show()

```

<!-- #region pycharm={"name": "#%% md\n"} -->
Although not perfect, these images certainly resemble MNIST digits!
Had we trained for longer and/or used a better model, we would certainly be able to
generate perfect-looking MNIST digits.
<!-- #endregion -->
