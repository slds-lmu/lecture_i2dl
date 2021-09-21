################################################################################ 
################################# preparations #################################
################################################################################ 
require("mxnet")
require("ggplot2")
require("reshape2")
setwd("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization/code")
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
dropoutInputValues = c(0, 0.2, 0.4, 0.6)
dropoutLayerValues = c(0, 0.1, 0.2, 0.3, 0.4, 0.5)
# empty data frame for the results with 100 rows (epochs) 
# and 5*5 columns (combinations of dropoutInput and dropoutLayer values)
resultsTrain = data.frame(matrix(ncol = length(dropoutInputValues)*length(dropoutLayerValues), nrow = epochs))
resultsTest = data.frame(matrix(ncol = length(dropoutInputValues)*length(dropoutLayerValues), nrow = epochs))

for(i in seq_along(dropoutInputValues)){
  for(j in seq_along(dropoutLayerValues)){
    print(paste0("Dropout Input Values ", i))
    print(paste0("Dropout Layer Values ", j))
    data = mx.symbol.Variable("data")
    drop0 = mx.symbol.Dropout(data = data, p = dropoutInputValues[i])
    fc1 = mx.symbol.FullyConnected(drop0, name = "fc1", num_hidden = neuronsLayer1)
    act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
    drop1 = mx.symbol.Dropout(data = act1, p = dropoutLayerValues[j])
    fc2 = mx.symbol.FullyConnected(drop1, name = "fc2", num_hidden = neuronsLayer2)
    act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
    drop2 = mx.symbol.Dropout(data = act2, p = dropoutLayerValues[j])
    fc3 = mx.symbol.FullyConnected(drop2, name = "fc3", num_hidden = neuronsLayer3)
    act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
    drop3 = mx.symbol.Dropout(data = act3, p = dropoutLayerValues[j])
    fc4 = mx.symbol.FullyConnected(drop3, name = "fc4", num_hidden = 10)
    softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
    devices = mx.cpu()
    mx.set.seed(7331)
    logger2 = mx.metric.logger$new()
    model = mx.model.FeedForward.create(softmax,
      X = train.x, y = train.y,
      eval.data = list(data = test.x, label = test.y),
      ctx = devices,
      learning.rate = learningRate,
      momentum = 0.9,
      num.round = epochs,
      array.batch.size = batchSize,
      eval.metric = mx.metric.accuracy,
      initializer = mx.init.uniform(0.07),
      epoch.end.callback = mx.callback.log.train.metric(5, logger2))
    resultsTrain[(i - 1)*length(dropoutLayerValues) + j] = as.numeric(lapply(logger2$train, function(x) 1-x))
    colnames(resultsTrain)[(i - 1)*length(dropoutLayerValues) + j] = paste("(", 
      dropoutInputValues[i], ";", dropoutLayerValues[j], ")", sep = "")
    resultsTest[(i - 1)*length(dropoutLayerValues) + j] = as.numeric(lapply(logger2$eval, function(x) 1-x))
    colnames(resultsTest)[(i - 1)*length(dropoutLayerValues) + j] = paste("(", 
      dropoutInputValues[i], ";", dropoutLayerValues[j], ")", sep = "")
  }
}    

################################################################################ 
################################## plot train ###################################
################################################################################ 
dropoutTrain = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(dropoutTrain)[1] = "epoch"

write.csv(dropoutTrain, file = "mnist_dropout_dropoutTrain_unmelted", row.names = FALSE, quote = FALSE)

mnist_dropout_dropoutTrain_unmelted = read.csv("mnist_dropout_dropoutTrain_unmelted", header = TRUE, check.names=FALSE)

dropoutTrain = melt(dropoutTrain, id = "epoch")

write.csv(dropoutTrain, file = "mnist_dropout_dropoutTrain", row.names = FALSE, quote = FALSE)

dropoutTrain = read.csv("mnist_dropout_dropoutTrain", header = TRUE)

ggplot(data = dropoutTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "error rate", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))
################################################################################ 
################################## plot test ###################################
################################################################################ 
dropoutTest = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(dropoutTest)[1] = "epoch"
write.csv(dropoutTest, file = "mnist_dropout_dropoutTest_unmelted", row.names = FALSE, quote = FALSE)

mnist_dropout_dropoutTrain_unmelted = read.csv("mnist_dropout_dropoutTrain_unmelted", header = TRUE, check.names=FALSE)
dropoutTest = mnist_dropout_dropoutTrain_unmelted[, -c(8:25)]

dropoutTest = melt(dropoutTest, id = "epoch")

write.csv(dropoutTest, file = "mnist_dropout_dropoutTest", row.names = FALSE, quote = FALSE)

dropoutTest = read.csv("mnist_dropout_dropoutTest", header = TRUE)

ggplot(data = dropoutTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "error rate", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))


