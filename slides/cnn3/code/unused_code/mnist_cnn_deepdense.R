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
batchSize = 300

results = data.frame(matrix(ncol = 2, nrow = epochs))

data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 30)
act1 = mx.symbol.Activation(data = conv1, act_type = "relu")
pool1 = mx.symbol.Pooling(data = act1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# second conv
conv2 = mx.symbol.Convolution(data = pool1,  kernel = c(3, 3), num_filter = 30)
act2 = mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 = mx.symbol.Pooling(data = act2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

flatten = mx.symbol.Flatten(data = pool2)

fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 1028)
act3 = mx.symbol.Activation(data = fc1, act_type = "relu")
fc2 = mx.symbol.FullyConnected(data = act3, num_hidden = 1028)
act4 = mx.symbol.Activation(data = fc2, act_type = "relu")
fc3 = mx.symbol.FullyConnected(data = act4, num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(data = fc3)

dim(train.x) = c(28, 28, 1, ncol(train.x))
dim(test.x) = c(28, 28, 1, ncol(test.x))

mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()

model = mx.model.FeedForward.create(softmax,
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices, 
  num.round = epochs, 
  array.batch.size = batchSize,
  learning.rate = 0.03, 
  momentum = 0.9, 
  wd = 0.001,
  initializer = mx.init.uniform(0.07),
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))

results[1] = as.numeric(lapply(logger$train, function(x) 1-x))
colnames(results)[1] = paste("Training Error")

results[2] = as.numeric(lapply(logger$eval, function(x) 1-x))
colnames(results)[2] = paste("Test Error")

################################################################################ 
################################# Plot Errors ##################################
################################################################################ 

cnnError = as.data.frame(cbind(1:dim(results)[1], results))
colnames(cnnError)[1] = "epoch"
cnnError = melt(cnnError, id = "epoch")

write.csv(cnnError, file = "cnnError", row.names = FALSE, quote = FALSE)

cnnError = read.csv("cnnError", header = TRUE)
options(scipen = 999)
cnnError$variable = factor(cnnError$variable)

ggplot(data = cnnError, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "misclassification", limits = c(0, 0.2)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "")