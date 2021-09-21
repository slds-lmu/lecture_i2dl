


#################### assign the location of the data as your wd()
train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)

train = data.matrix(train)
test = data.matrix(test)

###############################################
# Split data into matrix containing features and
# vector with labels
train.x = train[, -1]
train.y = train[, 1]

# normalize to (0,1) and transpose data
train.x = t(train.x/255)
dim(train.x)

test = t(test/255)

table(train.y)
###############################################

require("mxnet")

data = mx.symbol.Variable(name = "data")

layer1 = mx.symbol.FullyConnected(data = data, name = "layer1",
                                  num_hidden = 10L)
activation1 = mx.symbol.Activation(data = layer1, name = "activation1",
                                   act_type = "relu")
layer2 = mx.symbol.FullyConnected(data = activation1, name = "layer2",
                                  num_hidden = 10L)
activation2 = mx.symbol.Activation(data = layer2, name = "activation2",
                                   act_type = "relu")
layer3 = mx.symbol.FullyConnected(data = activation2, name = "layer3",
                                  num_hidden = 10L)
softmax = mx.symbol.SoftmaxOutput(data = layer3, name = "softmax")

###############################################

graph.viz(model$symbol)

##############################################

devices = mx.cpu()

mx.set.seed(1337)

model = mx.model.FeedForward.create(
  symbol = softmax,
  X = train.x, y = train.y,
  ctx = devices,
  num.round = 10L, array.batch.size = 100L,
  learning.rate = 0.05,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(100L))
###############################################

require("mxnet")

train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)
train = data.matrix(train)
test = data.matrix(test)
train.x = train[,-1]
train.y = train[,1]
train.x = t(train.x/255)
test = t(test/255)
data = mx.symbol.Variable("data")
layer1 = mx.symbol.FullyConnected(data, name = "layer1",num_hidden = 10)
activation1 = mx.symbol.Activation(layer1, name = "activation1", act_type = "relu")
layer2 = mx.symbol.FullyConnected(activation1, name = "layer2", num_hidden = 10)
activation2 = mx.symbol.Activation(layer2, name = "activation2", act_type = "relu")
layer3 = mx.symbol.FullyConnected(activation2, name = "layer3", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(layer3, name = "softmax")
devices = mx.cpu()
mx.set.seed(1337)
model = mx.model.FeedForward.create(softmax, X = train.x, y = train.y,
                                    ctx = devices, num.round = 10, array.batch.size = 100,
                                    learning.rate = 0.05, momentum = 0.9,
                                    eval.metric = mx.metric.accuracy,
                                    initializer = mx.init.uniform(0.07),
                                    epoch.end.callback = mx.callback.log.train.metric(100))

################################################

preds = predict(model, test)
# this yields us predicted probabilities for all 10 classes
dim(preds)

# we choose the maximum to obtain quantities for each class
pred.label = max.col(t(preds)) - 1
table(pred.label)

##############################################










