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
# Lab 10

**Authors**: Emilio Dorigatti, Tobias Weber

## Imports
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
from collections import Counter
from itertools import chain
import random
import shutil
import string
import urllib.request
from functools import reduce, partial
from math import ceil
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import nn, Tensor
from torch.distributions import Normal
from torch.optim import Adam, Optimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchtext.datasets import IMDB
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision.models import vgg16

set_matplotlib_formats('png', 'pdf')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
## Exercise 1

In this exercise, we are going to revise the sentiment classifier for IMDB reviews we
developed in a previous lab. Earlier, we encoded each review as a single "bag-of-words"
vector which had one element for each word in our dictionary set to one if that word was
found in the review, zero otherwise. This allowed us to use a simple fully-connected
neural network but, on the flip side, we lost all information contained in the ordering
and of the words and possible multiple repetitions. Recurrent neural networks, however,
are able to process reviews directly. Let's see how!

The first step is to load the data and preprocess it like it in exercise 6, so if you
still remember what we did there, feel free to skip this part.
For brevity, we only use the 10000 most common words
and truncate reviews to 250 words, but if you can use a GPU then feel free to
use the full length reviews and all words!
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
train_iterator, test_iterator = IMDB();
```

```python pycharm={"name": "#%%\n"}
# Feed iterator to list
train_x = []
train_y = []
test_x = []
test_y = []

for label, line in train_iterator:
    train_x.append(line)
    train_y.append(label)

for label, line in test_iterator:
    test_x.append(line)
    test_y.append(label)

```

```python pycharm={"name": "#%%\n"}
# Tokenize sentences.

def tokenize(data_list: List[str]) -> List[List[str]]:
    """
    Tokenize a list of strings.

    :param data_list: A list of strings.
    :return: A list where each entry is a list including the tokenized elements.
    """
    token_list: List[List[str]] = []
    for data_string in data_list:
        # Remove punctuation.
        data_string = data_string.translate(str.maketrans('', '', string.punctuation))
        # Split by space.
        token_list.append(data_string.split())
    return token_list

train_x = tokenize(train_x)
test_x = tokenize(test_x)

```

```python pycharm={"name": "#%%\n"}
# Count-vectorize sentences.
class CountVectorizer:
    def __init__(self):
        self.vec_to_str_map: Dict[int, str] = {}
        self.str_to_vec_map: Dict[str, int] = {}

    def fit(self, token_list: List[str]) -> None:
        # The `Counter` object from the `collections` library gives us efficient counting
        # in large lists out of box.
        cnt = Counter(token_list)
        sorted_cnt = sorted(cnt.items(), key=lambda item: item[1], reverse=True)
        sorted_words = [key for key, val in sorted_cnt]

        # Python does not know a bidirectional mapping by default.
        # We trick a bit by simply creating two dicts, but note that this is inefficient.
        self.str_to_vec_map = {sorted_words[i]: i + 1 for i in range(len(sorted_words))}
        self.vec_to_str_map = {i + 1: sorted_words[i] for i in range(len(sorted_words))}

    def transform_to_vec(self, token_list: List[str]) -> List[int]:
        return [self.str_to_vec_map.get(word) for word in token_list]

    def transform_to_str(self, token_list: List[int]) -> List[str]:
        return [self.vec_to_str_map.get(rank) for rank in token_list]

train_words = [word for word_list in train_x for word in word_list]
test_words = [word for word_list in test_x for word in word_list]

count_vectorizer = CountVectorizer()
counter = count_vectorizer.fit(train_words)

train_x = [count_vectorizer.transform_to_vec(word_list) for word_list in train_x]
test_x = [count_vectorizer.transform_to_vec(word_list) for word_list in test_x]
```

```python pycharm={"name": "#%%\n"}
# Discard words that are not in the top 10000
# Truncate sequences to a length of 250
# Remove Nones

def filter_word_ranks(
        word_list: List[Optional[int]],
        max_rank: int = 10000,
        max_seq_len: int = 250
) -> List[int]:
    output = []
    seq_len = 0
    for word_rank in word_list:
        if seq_len >= max_seq_len:
            return output
        elif word_rank is None:
            continue
        elif word_rank <= max_rank:
            output.append(word_rank)
            seq_len += 1
    return output

train_x = [filter_word_ranks(word_list) for word_list in train_x]
test_x = [filter_word_ranks(word_list) for word_list in test_x]
```

```python pycharm={"name": "#%%\n"}
# Encode labels to binary targets
train_y = [1 if label == 'pos' else 0 for label in train_y]
test_y = [1 if label == 'pos' else 0 for label in test_y]
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now, each review is a vector of numbers, each corresponding to a different word:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
print(train_x[0])
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Even though RNNs can process sequences of arbitrary length, all sequences in the same
batch must be of the same length, while sequences in different batches can have different
length. In this case, however, we pad all sequences to the same length as this makes for
much simpler code.
PyTorch provides a function to do so for you called `pad_sequence` (read the documentation!).
Hint: It might be good to set the argument `batch_first` to `True.
Beforehand, we need to convert the data to tensors. Let's also define our device and push
the newly created padded tensor to it.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
device = (
    #!TAG HWBEGIN
    #!MSG TODO: Define the device you want to use.
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #!TAG HWEND
)

train_x = [torch.tensor(word_list, dtype=torch.int) for word_list in train_x]
test_x = [torch.tensor(word_list, dtype=torch.int) for word_list in test_x]

train_x = (
    #!TAG HWBEGIN
    #!MSG TODO: Pad the sequences and push the result to your device.
    pad_sequence(train_x, batch_first=True).to(device)
    #!TAG HWEND
)

test_x = (
    #!TAG HWBEGIN
    #!MSG TODO: Pad the sequences and push the result to your device.
    pad_sequence(test_x, batch_first=True).to(device)
    #!TAG HWEND
)


train_y = torch.tensor(train_y, dtype=torch.float, device=device)
test_y = torch.tensor(test_y, dtype=torch.float, device=device)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
print(train_x.shape)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The data is now an array of shape `(num_samples x seq_len)`.
A PyTorch RNN with `batch_first=True` option expects the input to be of shape
`(num_samples x seq_len x features)`. Although we have a univariate timeseries, we still
need to add this additional last dimension.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
train_x = (
    #!TAG HWBEGIN
    #!MSG TODO: Add the feature dimension.
    train_x.unsqueeze(-1)
    #!TAG HWEND
)

test_x = (
    #!TAG HWBEGIN
    #!MSG TODO: Add the feature dimension.
    test_x.unsqueeze(-1)
    #!TAG HWEND
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Finally let's create our `IMDBDataset` object:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

class IMDBDataset(Dataset):
    def __init__(self, data: Tensor, labels: Tensor):
        self.data = data
        self.labels = labels

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        #!TAG HWBEGIN
        #!MSG TODO: Return the correct review and label for the index.
        return self.data[idx], self.labels[idx]
        #!TAG HWEND
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Next, we define our sequential model. The first layer is an _Embedding_ layer that
associates a vector of numbers to each word in the vocabulary. These numbers are updated
during training just like all other weights in the network. Crucially, thanks to this
embedding layer we do not have to one-hot-encode the reviews but we can use the word
indices directly, making the process much more efficient.

Note the parameter `padding_idx`: this indicates that zeros in the input sequences are
used for padding (verify that this is the case!). Internally, this is used by the RNN to
ignore padding tokens, preventing them from contributing to the gradients (read more in
the user guide, [link](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html/)!)

We also make use of a `nn.Module` container, where we will define our model. This
gives us more flexibility in the flow of the network. Here we add the model blocks
as class attributes and define a forward pass, which is enough for the autograd engine.

The shapes and dimensions of tensors can now be a bit tricky. It helps if you print
the resulting shape of each transformation to console and investigate what happened!
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
class LSTMModel(nn.Module):
    def __init__(self):
        # A class that inherits from nn.Module needs to call the constructor from the
        # parent class
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=10001,
            embedding_dim=64,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # The output needs to be reshaped or otherwise we have a dimension too much.
        x = self.embedding(x).squeeze(2)

        # The LSTM module gives a variety of outputs. Please refer to the official
        # docs for a detailed description. Here `hidden` contains the final hidden states
        # from the last layer for every sample in the batch.
        _, (hidden, _) = self.lstm(x)

        # We need to extract the last hidden state
        y_score = self.fc(hidden[-1])
        y_hat = self.sigmoid(y_score).squeeze(-1)

        return y_hat
```

<!-- #region pycharm={"name": "#%% md\n"} -->
In the next step, we once again need our beloved training loop.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def train(
        model: nn.Module,
        loss: nn.Module,
        optimizer: Optimizer,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epochs: int,
        batch_size: int
) -> Dict:

    metrics: Dict = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    num_train_batches = ceil(len(train_dataset) / batch_size)
    num_test_batches = ceil(len(test_dataset) / batch_size)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)


    for ep in range(1, epochs + 1):
        total_loss = 0
        num_correct = 0

        ################################################################################
        # TRAINING LOOP
        ################################################################################

        for batch_idx, (x, y) in enumerate(train_loader):

            #!TAG HWBEGIN
            #!MSG TODO: Add forward pass + batch loss, backpropagation and apply gradients
            y_hat = model(x)
            batch_loss = loss(y_hat, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            #!TAG HWEND

            if batch_idx % 10 == 0:
                print('TRAINING BATCH:\t({:5} / {:5})\tLOSS:\t{:.3f}'
                      .format(batch_idx, num_train_batches, float(batch_loss)), end='\r')

            total_loss += float(batch_loss)
            num_correct += int(torch.sum(torch.where(y_hat > 0.5, 1, 0) == y))


        ep_train_loss = total_loss / len(train_dataset)
        ep_train_acc = num_correct / len(train_dataset)

        total_loss = 0
        num_correct = 0

        ################################################################################
        # TEST LOOP
        ################################################################################

        for batch_idx, (x, y) in enumerate(test_loader):

            with torch.no_grad():
                #!TAG HWBEGIN
                #!MSG TODO: Do a forward pass and get the batch loss
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                #!TAG HWEND

            if batch_idx % 50 == 0:
                print('TEST BATCH:\t({:5} / {:5})\tLOSS:\t{:.3f}'
                      .format(batch_idx, num_test_batches, float(batch_loss)), end='\r')

            total_loss += float(batch_loss)
            num_correct += int(torch.sum(torch.where(y_hat > 0.5, 1, 0) == y))

        ep_test_loss = total_loss / len(test_dataset)
        ep_test_acc = num_correct / len(test_dataset)

        metrics['train_loss'].append(ep_train_loss)
        metrics['train_acc'].append(ep_train_acc)
        metrics['test_loss'].append(ep_test_loss)
        metrics['test_acc'].append(ep_test_acc)

        print('EPOCH:\t{:5}\tTRAIN LOSS:\t{:.3f}\tTRAIN ACCURACY:\t{:.3f}\tTEST LOSS:\t'
              '{:.3f}\tTEST ACCURACY:\t{:.3f}'
              .format(ep, ep_train_loss, ep_train_acc,ep_test_loss, ep_test_acc,
                      end='\r'))
    return metrics
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We declare model, optimizer, datasets, loss, epochs, batch size and then train!
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
epochs = 15
batch_size = 32

model = (
    #!TAG HWBEGIN
    #!MSG Initialize the model and push it to your device.
    LSTMModel().to(device)
    #!TAG HWEND
)

optimizer = (
    #!TAG HWBEGIN
    #!MSG TODO: Define an optimizer.
    Adam(model.parameters(), lr=1e-2)
    #!TAG HWEND
)

loss = (
    #!TAG HWBEGIN
    #!MSG TODO: Define the matching loss function.
    nn.BCELoss()
    #!TAG HWEND
)

train_dataset = (
    #!TAG HWBEGIN
    #!MSG TODO: Define the train dataset.
    IMDBDataset(train_x, train_y)
    #!TAG HWEND
)

test_dataset = (
    #!TAG HWBEGIN
    #!MSG TODO: Define the train dataset.
    IMDBDataset(test_x, test_y)
    #!TAG HWEND
)


metrics = train(model, loss, optimizer, train_dataset, test_dataset, epochs, batch_size)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def get_training_progress_plot(
        train_losses: List[float],
        train_accs: List[float],
        val_losses: List[float],
        val_accs: List[float],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2))

    ax1.set_title('Loss')
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Test Loss')
    ax1.legend()

    ax2.set_title('Accuracy')
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Test Accuracy')
    ax2.legend()

get_training_progress_plot(
    metrics['train_loss'],
    metrics['train_acc'],
    metrics['test_loss'],
    metrics['test_acc'],
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The model seems to be learning more easily than the simple baseline we created time ago,
which had an accuracy of 85-88% on the test data.
Let it train for longer and tune the
architecture above to reach as high accuracy as possible! (note that evaluating on the
same data that you used for early stopping is cheating).


## Exercise 2

In this exercise, we are going to implement an autoencoder and train it on the MNIST
dataset.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
train_x = MNIST(root='.data', download=True, transform=ToTensor());
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As an introduction, we will train PCA, i.e. an autoencoder with linear encoder
$\textbf{z}=\textbf{Ex}$ and linear decoder: $\textbf{x'}=\textbf{Dz}$.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We will flatten the image to a vector
img_features = 28*28

# This sets the size of our latent space
latent_size = 8

encoder = (
    #!TAG HWBEGIN
    #!MSG Define a linear encoder on your device.
    nn.Linear(in_features=img_features, out_features=latent_size).to(device)
    #!TAG HWEND
)

decoder = (
    #!TAG HWBEGIN
    #!MSG Define a linear decoder on your device.
    nn.Linear(in_features=latent_size, out_features=img_features).to(device)
    #!TAG HWEND
)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As usual we define a training loop:
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def train_autoencoder(
        encoder: nn.Module,
        decoder: nn.Module,
        loss: nn.Module,
        optimizer: Optimizer,
        mnist_dataset: MNIST,
        epochs: int,
        batch_size: int,
        device: torch.device
) -> List[float]:

    train_losses = []

    num_train_batches = ceil(len(mnist_dataset) / batch_size)
    train_loader = DataLoader(mnist_dataset, batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        total_loss = 0

        ################################################################################
        # TRAINING LOOP
        ################################################################################

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device).view(x.shape[0], -1)

            #!TAG HWBEGIN
            #!MSG TODO: Use Encoder/Decoder to compress/reconstruct an image
            #!MSG TODO: Compute the reconstruction batch_loss per sample
            #!MSG TODO: Do the backward pass
            x_hat = decoder(encoder(x))
            batch_loss = loss(x, x_hat) / batch_size

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            #!TAG HWEND

            if batch_idx % 10 == 0:
                print('TRAINING BATCH:\t({:5} / {:5})\tLOSS:\t{:.3f}'
                      .format(batch_idx, num_train_batches, float(batch_loss)), end='\r')

            total_loss += float(batch_loss)

        train_losses.append(total_loss / num_train_batches)
        print('EPOCH:\t{:5}\tTRAIN LOSS:\t{:.3f}'.format(ep, train_losses[-1], end='\r'))

    return train_losses
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Next, we initialize a loss function, optimizer and train for 1 epoch.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
epochs = 1
batch_size = 32

optimizer = (
    #!TAG HWBEGIN
    #!MSG TODO: Define an optimizer.
    # #!MSG Hint: Use `chain` from itertools to add multiple modules to an optimizer at once.
    Adam(chain(encoder.parameters(), decoder.parameters()))
    #!TAG HWEND
)

loss = (
    #!TAG HWBEGIN
    #!MSG TODO: Define the reconstruction loss.
    #!MSG Hint: Use the `reduction = 'sum'` argument.
    nn.MSELoss(reduction='sum')
    #!TAG HWEND
)

train_autoencoder(encoder, decoder, loss, optimizer, train_x, epochs, batch_size, device)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Let's inspect the reconstructions quality for a few samples.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

def plot_reconstruction_grid(
        encoder: nn.Module, decoder: nn.Module, mnist_dataset: MNIST) -> None:
    # We have a bit of struggle with converting from devices, ranges and different shapes here.
    # All this could have been prevented, if all our modules would be packaged in
    # appropriate classes and utility methods.
    x_samples = mnist_dataset.data[:100] / 255
    x_hat_samples = decoder(encoder(x_samples.to(device).view(100, -1)))
    x_hat_samples = torch.clamp(x_hat_samples.detach().cpu().view(100, 28, 28), 0, 1)

    cur_col = 0
    image_list = []
    for _ in range(4):
        image_list.extend(x_samples[cur_col:cur_col + 25])
        image_list.extend(x_hat_samples[cur_col:cur_col + 25])
        cur_col += 25

    image_batch = torch.stack(image_list).unsqueeze(1)
    image_grid = make_grid(image_batch, nrow=25)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

plot_reconstruction_grid(encoder, decoder, train_x)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
As you can see, reducing the whole image to a latent space of 8 combined with a rather
short training of 1 epoch results in constructions that lack detail and are far from
perfect.
In the next step, we will define a more complex non-linear encoder/decoder and train
a few epochs more.
Try to have several dense layers and experiment with different activation functions.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC
epochs = 5
batch_size = 64

encoder = nn.Sequential(
    #!TAG HWBEGIN
    #!MSG Define an encoder.
    nn.Linear(in_features=img_features, out_features=128),
    nn.LeakyReLU(),
    nn.Linear(in_features=128, out_features=64),
    nn.LeakyReLU(),
    nn.Linear(in_features=64, out_features=latent_size)
    #!TAG HWEND
).to(device)

decoder = nn.Sequential(
    #!TAG HWBEGIN
    #!MSG Define a decoder.
    nn.Linear(in_features=latent_size, out_features=64),
    nn.LeakyReLU(),
    nn.Linear(in_features=64, out_features=128),
    nn.LeakyReLU(),
    nn.Linear(in_features=128, out_features=img_features)
    #!TAG HWEND
).to(device)

optimizer = (
    #!TAG HWBEGIN
    #!MSG TODO: Define an optimizer.
    # #!MSG Hint: Use `chain` from itertools to add multiple modules to an optimizer at once.
    Adam(chain(encoder.parameters(), decoder.parameters()))
    #!TAG HWEND
)

loss = (
    #!TAG HWBEGIN
    #!MSG TODO: Define the reconstruction loss.
    #!MSG Hint: Use the `reduction = 'sum'` argument.
    nn.MSELoss(reduction='sum')
    #!TAG HWEND
)

losses = train_autoencoder(encoder, decoder, loss, optimizer, train_x, epochs, batch_size, device)
```

```python pycharm={"name": "#%%\n"}
#!TAG SKIPQUESTEXEC

plot_reconstruction_grid(encoder, decoder, train_x)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The reconstructions look way better now! Further training will increase the image
quality. Keep in mind that autoencoders (like most DL architectures) are prone to
overfitting. The quality of reconstructions does not automatically coincide with
good representations.
<!-- #endregion -->
