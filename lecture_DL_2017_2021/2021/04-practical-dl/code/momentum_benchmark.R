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
learningRate = 0.03
neuronsLayer1 = 512
neuronsLayer2 = 512
neuronsLayer3 = 512
mom =  c(0, 0.5, 0.9, 0.99)
################################################################################ 
#################### architecture 1 without dropout ############################
################################################################################ 
resultsTrain = data.frame(matrix(ncol = length(mom), nrow = epochs))
resultsTest = data.frame(matrix(ncol = length(mom), nrow = epochs))

for(i in seq_along(mom)){
  print(paste0("momentum ", i))
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
  model = mx.model.FeedForward.create(softmax,
    X = train.x, y = train.y,
    eval.data = list(data = test.x, label = test.y),
    ctx = devices,
    learning.rate = learningRate,
    momentum = mom[i],
    wd = 0.001,
    num.round = epochs,
    array.batch.size = batchSize,
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback = mx.callback.log.train.metric(5, logger))
  resultsTrain[i] = as.numeric(lapply(logger$train, function(x) 1-x))
  colnames(resultsTrain)[i] = paste("momentum ", 
    mom[i], sep = "")
  resultsTest[i] = as.numeric(lapply(logger$eval, function(x) 1-x))
  colnames(resultsTest)[i] = paste("momentum ", 
    mom[i], sep="")
}

################################################################################ 
################################## plot train ##################################
################################################################################ 
momentumTrain = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(momentumTrain)[1] = "epoch"
momentumTrain = melt(momentumTrain, id = "epoch")

write.csv(momentumTrain, file = "momentumTrain", row.names = FALSE, quote = FALSE)

momentumTrain = read.csv("momentumTrain", header = TRUE)
options(scipen=999)
momentumTrain$variable = factor(momentumTrain$variable)

ggplot(data = momentumTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 0.2)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "momentum")
################################################################################ 
################################## plot test ###################################
################################################################################ 
momentumTest = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(momentumTest)[1] = "epoch"
momentumTest = melt(momentumTest, id = "epoch")

write.csv(momentumTest, file = "momentumTest", row.names = FALSE, quote = FALSE)

momentumTest = read.csv("momentumTest", header = TRUE)
options(scipen=999)
momentumTest$variable = factor(momentumTest$variable)

ggplot(data = momentumTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.2)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "momentum")
