################################################################################ 
################################# preparations #################################
################################################################################ 
require("mxnet")
require("ggplot2")
require("reshape2")
setwd("C:/Users/Niklas/lectures/deeplearning/2017/03-optimization/code")
# train data: 5000 rows with 28*28 = 784 + 1 (label) columns
train = read.csv("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization/train_short.csv", header = TRUE)
# test data: 1000 rows with 28*28 = 784 + 1 (label) columns
test = read.csv("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization/test_short.csv", header = TRUE)
train = data.matrix(train)
test = data.matrix(test)
# remove labels from train data, transpose and normalize
train.x = t(train[, -1]/255)
# vector with training labels
train.y = train[, 1]
# remove labels from test data, transpose and normalize
test.x = t(test[, -1]/255)
# test labels
test.y = test[, 1]
################################################################################ 
############################## global variables ################################
################################################################################ 
epochs = 100
batchSize = 100
neuronsLayer1 = 512
neuronsLayer2 = 512
neuronsLayer3 = 512
resultsTrain = data.frame(matrix(ncol = 4, nrow = epochs))
resultsTest = data.frame(matrix(ncol = 4, nrow = epochs))
################################################################################ 
######################## architecture 1 with sgd ###############################
################################################################################ 
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name = "fc1", num_hidden = neuronsLayer1)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 = mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = neuronsLayer2)
act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 = mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = neuronsLayer3)
act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
fc4 = mx.symbol.FullyConnected(act3, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
devices = mx.cpu()
mx.set.seed(7331)
logger = mx.metric.logger$new()
modelSGD = mx.model.FeedForward.create(softmax,
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  optimizer = "sgd",
  learning.rate = 0.03,
  wd = 0.001,
  momentum = 0.9,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
resultsTrain[1] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(resultsTrain)[1] = paste("SGD ")
resultsTest[1] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(resultsTest)[1] = paste("SGD ")
################################################################################ 
####################### architecture 2 with Adagrad ############################
################################################################################ 
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name = "fc1", num_hidden = neuronsLayer1)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 = mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = neuronsLayer2)
act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 = mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = neuronsLayer3)
act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
fc4 = mx.symbol.FullyConnected(act3, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
devices = mx.cpu()
mx.set.seed(7331)
logger = mx.metric.logger$new()
modelSGD = mx.model.FeedForward.create(softmax,
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  optimizer = "adagrad",
  wd = 0.001,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
resultsTrain[2] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(resultsTrain)[2] = paste("Adagrad")
resultsTest[2] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(resultsTest)[2] = paste("Adagrad")
################################################################################ 
####################### architecture 3 with RMSProp ############################
################################################################################ 
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name = "fc1", num_hidden = neuronsLayer1)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 = mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = neuronsLayer2)
act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 = mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = neuronsLayer3)
act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
fc4 = mx.symbol.FullyConnected(act3, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
devices = mx.cpu()
mx.set.seed(7331)
logger = mx.metric.logger$new()
modelSGD = mx.model.FeedForward.create(softmax,
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  optimizer = "rmsprop",
  wd = 0.001,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
resultsTrain[3] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(resultsTrain)[3] = paste("RMSProp")
resultsTest[3] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(resultsTest)[3] = paste("RMSProp")
################################################################################ 
######################## architecture 4 with Adam ##############################
################################################################################
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name = "fc1", num_hidden = neuronsLayer1)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 = mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = neuronsLayer2)
act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 = mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = neuronsLayer3)
act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
fc4 = mx.symbol.FullyConnected(act3, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
devices = mx.cpu()
mx.set.seed(7331)
logger = mx.metric.logger$new()
modelSGD = mx.model.FeedForward.create(softmax,
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  optimizer = "adam",
  wd = 0.001,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
resultsTrain[4] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(resultsTrain)[4] = paste("Adam")
resultsTest[4] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(resultsTest)[4] = paste("Adam")
################################################################################ 
################################## plot train ##################################
################################################################################ 
optimizerTrain = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(optimizerTrain)[1] = "epoch"
optimizerTrain = melt(optimizerTrain, id = "epoch")

write.csv(optimizerTrain, file = "optimizerTrain", row.names = FALSE, quote = FALSE)

optimizerTrain = read.csv("optimizerTrain", header = TRUE)
options(scipen=999)
optimizerTrain$variable = factor(optimizerTrain$variable)

ggplot(data = optimizerTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 0.25)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "Optimizer")
################################################################################ 
################################## plot test ###################################
################################################################################ 
optimizerTest = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(optimizerTest)[1] = "epoch"
optimizerTest = melt(optimizerTest, id = "epoch")

write.csv(optimizerTest, file = "optimizerTest", row.names = FALSE, quote = FALSE)

optimizerTest = read.csv("optimizerTest", header = TRUE)
options(scipen=999)
optimizerTest$variable = factor(optimizerTest$variable)

ggplot(data = optimizerTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.25)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "Optimizer")