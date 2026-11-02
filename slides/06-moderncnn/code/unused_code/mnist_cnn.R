################################################################################ 
################################# preparations #################################
################################################################################ 
require("mxnet")
require("ggplot2")
require("reshape2")
setwd("C:/Users/Niklas/lectures/deeplearning/2017/04-cnns/code")
train = read.csv("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization/train_short.csv", header = TRUE)
test = read.csv("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization/test_short.csv", header = TRUE)
train = data.matrix(train)
test = data.matrix(test)
train.x = train[,-1]
train.y = train[,1]
train.x = t(train.x/255)
test.x = t(test[, -1]/255)
test.y = test[, 1]
################################################################################ 
############################## global variables ################################
################################################################################
epochs = 100
batchSize = 100
filterLayer1 = 20
filterLayer2 = 50
filterLayer1Size = c(5, 5)
filterLayer2Size = c(5, 5)
neuronsLayer1 = 512
neuronsLayer2 = 10
resultsTrain = data.frame(matrix(ncol = 2, nrow = epochs))
resultsTest = data.frame(matrix(ncol = 2, nrow = epochs))
################################################################################ 
############################ architecture relu #################################
################################################################################ 
data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data = data, kernel = filterLayer1Size, num_filter = filterLayer1)
tanh1 = mx.symbol.Activation(data = conv1, act_type = "relu")
pool1 = mx.symbol.Pooling(data = tanh1, pool_type = "max",
  kernel = c(2, 2), stride = c(2, 2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel = filterLayer2Size, num_filter = filterLayer2)
tanh2 = mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 = mx.symbol.Pooling(data = tanh2, pool_type = "max",
  kernel = c(2, 2), stride = c(2, 2))
# first dense
flatten = mx.symbol.Flatten(data = pool2)
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = neuronsLayer1)
tanh3 = mx.symbol.Activation(data = fc1, act_type = "relu")
# second dense
fc2 = mx.symbol.FullyConnected(data = tanh3, num_hidden = neuronsLayer2)
# loss function
lenet = mx.symbol.SoftmaxOutput(data = fc2)
# flatten data
train.array = train.x
dim(train.array) = c(28, 28, 1, ncol(train.x))
test.array = test.x
dim(test.array) = c(28, 28, 1, ncol(test.x))
mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()
# hyperparameters
model = mx.model.FeedForward.create(lenet,
  X = train.array, y = train.y,
  eval.data = list(data = test.array, label = test.y),
  ctx = devices, 
  num.round = epochs, 
  array.batch.size = batchSize,
  learning.rate = 0.03, 
  momentum = 0.9, 
  wd = 0.001,
  initializer = mx.init.uniform(0.07),
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))

resultsTrain[1] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(resultsTrain)[1] = paste("relu")
resultsTest[1] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(resultsTest)[1] = paste("relu")
################################################################################
############################# architecture tanh ################################
################################################################################ 
data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data = data, kernel = filterLayer1Size, num_filter = filterLayer1)
tanh1 = mx.symbol.Activation(data = conv1, act_type = "tanh")
pool1 = mx.symbol.Pooling(data = tanh1, pool_type = "max",
  kernel = c(2, 2), stride = c(2, 2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel = filterLayer2Size, num_filter = filterLayer2)
tanh2 = mx.symbol.Activation(data = conv2, act_type = "tanh")
pool2 = mx.symbol.Pooling(data = tanh2, pool_type = "max",
  kernel = c(2, 2), stride = c(2, 2))
# first dense
flatten = mx.symbol.Flatten(data = pool2)
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = neuronsLayer1)
tanh3 = mx.symbol.Activation(data = fc1, act_type = "tanh")
# second dense
fc2 = mx.symbol.FullyConnected(data = tanh3, num_hidden = neuronsLayer2)
# loss function
lenet = mx.symbol.SoftmaxOutput(data = fc2)
# flatten data
train.array = train.x
dim(train.array) = c(28, 28, 1, ncol(train.x))
test.array = test.x
dim(test.array) = c(28, 28, 1, ncol(test.x))
mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()
# hyperparameters
model = mx.model.FeedForward.create(lenet,
  X = train.array, y = train.y,
  eval.data = list(data = test.array, label = test.y),
  ctx = devices, 
  num.round = epochs, 
  array.batch.size = batchSize,
  learning.rate = 0.03, 
  momentum = 0.9, 
  wd = 0.001,
  initializer = mx.init.uniform(0.07),
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))

resultsTrain[2] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(resultsTrain)[2] = paste("tanh")
resultsTest[2] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(resultsTest)[2] = paste("tanh")
################################################################################ 
################################## plot train ##################################
################################################################################ 
cnnTrain = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(cnnTrain)[1] = "epoch"
cnnTrain = melt(cnnTrain, id = "epoch")

write.csv(cnnTrain, file = "cnnTrain", row.names = FALSE, quote = FALSE)

cnnTrain = read.csv("cnnTrain", header = TRUE)
options(scipen=999)
cnnTrain$variable = factor(cnnTrain$variable)

ggplot(data = cnnTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "Activation")
################################################################################ 
################################## plot test ###################################
################################################################################ 
cnnTest = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(cnnTest)[1] = "epoch"
cnnTest = melt(cnnTest, id = "epoch")

write.csv(cnnTest, file = "cnnTest", row.names = FALSE, quote = FALSE)

cnnTest = read.csv("cnnTest", header = TRUE)
options(scipen=999)
cnnTest$variable = factor(cnnTest$variable)

ggplot(data = cnnTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "Activation")




