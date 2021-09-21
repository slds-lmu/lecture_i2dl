library(mxnet)

# Fix related seeds
mx.set.seed(1337)

setwd("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization")

# Read MNIST data downloaded from Kaggle
rawdata = read.csv("train_short.csv", header = T)
rawdata = as.matrix(rawdata)

# Divide into training data and test data
train.index = sample(x = 1:nrow(rawdata), size = 4000)
train = rawdata[train.index, ]
test = rawdata[-train.index, ]

# Divide into data and labels. To transpose it to colmajor form
train.x = t(train[, -1]/ 255)
train.y = train[,  1]
test.x = t(test[, -1]/ 255)
test.y = test[,  1]

# Create an autoencoder that compresses nrow (input) dimension data into hidden_size dimensions
pretrain = function(input, hidden_size)
{
  input_size = nrow(input)  # colmajor

  # Define autoencoder
  symbol = mx.symbol.Variable("data")
  symbol = mx.symbol.FullyConnected(data = symbol, name = "encoder", num_hidden = hidden_size)
  symbol = mx.symbol.Activation(data = symbol, name = "encoder_act", act_type = "relu")
  symbol = mx.symbol.FullyConnected(data = symbol, name = "decoder", num_hidden = input_size)
  symbol = mx.symbol.LinearRegressionOutput(data = symbol, name = "output")

  # learn
  model = mx.model.FeedForward.create(
    symbol = symbol,
    X = input, y = input,
    ctx = mx.cpu(),
    num.round = 10, array.batch.size = 100,
    optimizer = "sgd", learning.rate = 0.01, momentum = 0.9,
    initializer = mx.init.Xavier(rnd_type = "uniform", factor_type = "in", magnitude = 2),
    eval.metric = mx.metric.rmse,
    batch.end.callback = mx.callback.log.train.metric(10),
    array.layout = "colmajor")

  return(model)
}

# To learn the next layer, calculate the output value of the current layer
encode = function(input, model)
{
  # Save learned params
  arg.params = model$arg.params[c("encoder_weight", "encoder_bias")]

  # define encoder
  symbol = mx.symbol.Variable("data")
  symbol = mx.symbol.FullyConnected(data = symbol, name = "encoder", num_hidden = ncol(arg.params$encoder_weight))
  symbol = mx.symbol.Activation(data = symbol, name = "encoder_act", act_type = "relu")

  # use learned params
  model = list(symbol = symbol, arg.params = arg.params, aux.params = list())
  class(model) = "MXFeedForwardModel"

  
  output = predict(model, input, array.layout = "colmajor")

  return(output)
}

# Learning each layer using pretrain and encode defined above
input.1 = train.x
model.1 = pretrain(input = input.1, hidden_size = 392)
input.2 = encode(input = input.1, model = model.1)
model.2 = pretrain(input = input.2, hidden_size = 196)
input.3 = encode(input = input.2, model = model.2)
model.3 = pretrain(input = input.3, hidden_size = 98)
input.4 = encode(input = input.3, model = model.3)
model.4 = pretrain(input = input.4, hidden_size = 49)

# Combine each encoder and define a 10 class classifier
symbol = mx.symbol.Variable("data")
symbol = mx.symbol.FullyConnected(data = symbol, name = "encoder_1", num_hidden = 392)
symbol = mx.symbol.Activation(data = symbol, name = "encoder_act_1", act_type = "relu")
symbol = mx.symbol.FullyConnected(data = symbol, name = "encoder_2", num_hidden = 196)
symbol = mx.symbol.Activation(data = symbol, name = "encoder_act_2", act_type = "relu")
symbol = mx.symbol.FullyConnected(data = symbol, name = "encoder_3", num_hidden = 98)
symbol = mx.symbol.Activation(data = symbol, name = "encoder_act_3", act_type = "relu")
symbol = mx.symbol.FullyConnected(data = symbol, name = "encoder_4", num_hidden = 49)
symbol = mx.symbol.Activation(data = symbol, name = "encoder_act_4", act_type = "relu")
symbol = mx.symbol.FullyConnected(data = symbol, name = "affine", num_hidden = 10)
symbol = mx.symbol.SoftmaxOutput(data = symbol, name = "output")

# Next, we extract the learned parameters of each autoencoder and rename 
# them for 10 class classifiers. The name seems to be a weight of <layer name> _weight, 
# bias of <layer name> _bias. It seems that layer and parameter are linked by name, and 
# if name is wrong, please note that "Parameter not found" error occurs.

# Retrieve the learned parameters
arg.params = list()
arg.params = c(arg.params, "encoder_1_weight" = model.1$arg.params$encoder_weight)
arg.params = c(arg.params, "encoder_1_bias" = model.1$arg.params$encoder_bias)
arg.params = c(arg.params, "encoder_2_weight" = model.2$arg.params$encoder_weight)
arg.params = c(arg.params, "encoder_2_bias" = model.2$arg.params$encoder_bias)
arg.params = c(arg.params, "encoder_3_weight" = model.3$arg.params$encoder_weight)
arg.params = c(arg.params, "encoder_3_bias" = model.3$arg.params$encoder_bias)
arg.params = c(arg.params, "encoder_4_weight" = model.4$arg.params$encoder_weight)
arg.params = c(arg.params, "encoder_4_bias" = model.4$arg.params$encoder_bias)

# We initialize the parameters of the newly added layer on our own. 
# When initializing all parameters, the initializer specified in the argument of 
# mx.model.FeedForward.create initializes, but if you want to initialize partly 
# like this time, It seems that it seems necessary to perform conversion 
# (see the implementation of mx.model.FeedForward.create, it initializes with 
# initializer, but if arg.params was specified, the process of entirely replacing 
# with the object , So arg.params seems to have to store all necessary parameters initialized).

# Finally, we will do learning again through the end.
model = mx.model.FeedForward.create(
  symbol = symbol,
  X = train.x, y = train.y,
  ctx = mx.cpu(),
  num.round = 20, array.batch.size = 100,
  optimizer = "sgd", learning.rate = 0.01, momentum = 0.9,
  eval.metric = mx.metric.accuracy,
  batch.end.callback = mx.callback.log.train.metric(10),
  array.layout = "colmajor",
  arg.params = arg.params, aux.params = NULL)

# Confirm prediction result

score = predict(model, test.x, array.layout = "colmajor")
label = max.col(t(score)) - 1
table(test.y, label)









