################################################################################ 
################################# preparations #################################
################################################################################ 
require("mxnet")
require("ggplot2")
require("reshape2")
setwd("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization")
train = read.csv("train_short.csv", header = TRUE)
test = read.csv("test_short.csv", header = TRUE)
train = data.matrix(train)
test = data.matrix(test)
train.x = t(train[, -1]/255)
train.y = train[, 1]
test.x = t(test[, -1]/255)
test.y = test[, 1]
################################################################################ 
############################## global variables ################################
################################# benchmark 1 ##################################
################################################################################ 
epochs = 200
batchSize = 100
learningRate = 0.03
neuronsLayer1 = 512
neuronsLayer2 = 512
neuronsLayer3 = 512
# values to iterate
weightDecay = c(0.01, 0.001, 0.0001, 0.00001, 0)

# empty data frames for the results with 100 rows (epochs) 
# and 6 rows (weightDecay values)
resultsTrain = data.frame(matrix(ncol = length(weightDecay), nrow = epochs))
resultsTest = data.frame(matrix(ncol = length(weightDecay), nrow = epochs))

for(i in seq_along(weightDecay)){
  print(paste0("Dropout Layer Values ", i))
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
    momentum = 0.9,
    wd = weightDecay[i],
    num.round = epochs,
    array.batch.size = batchSize,
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback = mx.callback.log.train.metric(5, logger))
  resultsTrain[i] = as.numeric(lapply(logger$train, function(x) 1-x))
  colnames(resultsTrain)[i] = paste("wd ", 
    weightDecay[i], sep = "")
  resultsTest[i] = as.numeric(lapply(logger$eval, function(x) 1-x))
  colnames(resultsTest)[i] = paste("wd ", 
    weightDecay[i], sep="")
}
    

################################################################################ 
################################## plot train ###################################
################################################################################ 
wdTrain = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(wdTrain)[1] = "epoch"
wdTrain = melt(wdTrain, id = "epoch")

write.csv(wdTrain, file = "wdTrain", row.names = FALSE, quote = FALSE)

wdTrain = read.csv("code/mnist_weight_decay_wdTrain", header = TRUE)
options(scipen=999)
wdTrain$variable = factor(wdTrain$variable)

ggplot(data = wdTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "weight decay")
################################################################################ 
################################## plot test ###################################
################################################################################ 
wdTest = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(wdTest)[1] = "epoch"
wdTest = melt(wdTest, id = "epoch")

write.csv(wdTest, file = "wdTest", row.names = FALSE, quote = FALSE)

wdTest = read.csv("code/mnist_weight_decay_wdTest", header = TRUE)
options(scipen=999)
wdTest$variable = factor(wdTest$variable)

ggplot(data = wdTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "weight decay")

