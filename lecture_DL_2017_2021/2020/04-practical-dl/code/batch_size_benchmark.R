################################################################################ 
#################### Deep Neural Network with Dropout ##########################
################################################################################

# Set your working directory to save the results from the computations
setwd("../03-optimization/code")

require("mxnet")
require("ggplot2")
require("reshape2")

epochs = 40
batchSize = c(16, 17, 20, 100, 128, 1000)

resultsTrain = data.frame(matrix(ncol = length(batchSize),
  nrow = epochs))
resultsTest = data.frame(matrix(ncol = length(batchSize), 
  nrow = epochs))
resultsTime = numeric(length(batchSize))


for(i in seq_along(batchSize)){
  
  print(paste0("Batchsize iteration ", i, " of ", length(batchSize)))
  
  data = mx.symbol.Variable("data")
  fc1 = mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128)
  act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
  fc2 = mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
  act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
  fc3 = mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 32)
  act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
  fc4 = mx.symbol.FullyConnected(act3, name = "fc4", num_hidden = 10)
  softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
  
  devices = mx.cpu()
  mx.set.seed(1337)
  logger = mx.metric.logger$new()
  
  timeBegin = Sys.time()
  model = mx.model.FeedForward.create(symbol = softmax,
    X = train.x, y = train.y,
    eval.data = list(data = test.x, label = test.y),
    ctx = devices,
    optimizer = "sgd",
    learning.rate = 0.05,
    momentum = 0.9,
    wd = 0.001,
    num.round = epochs,
    array.batch.size = batchSize[i],
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.uniform(0.07),
    epoch.end.callback = mx.callback.log.train.metric(5, logger))
  timeEnd = Sys.time()
  
  resultsTrain[i] = as.numeric(lapply(logger$train, function(x) 1-x))
  colnames(resultsTrain)[i] = paste(batchSize[i])
  
  resultsTest[i] = as.numeric(lapply(logger$eval, function(x) 1-x))
  colnames(resultsTest)[i] = paste(batchSize[i])
  
  resultsTime[i] = timeEnd - timeBegin
  
}

################################################################################ 
############################ Plot Training Errors ##############################
################################################################################ 

nnBenchmarkTrainError = as.data.frame(cbind(1:dim(resultsTrain)[1], resultsTrain))
colnames(nnBenchmarkTrainError)[1] = "epoch"
nnBenchmarkTrainError = melt(nnBenchmarkTrainError, id = "epoch")

write.csv(nnBenchmarkTrainError, file = "nnBenchmarkTrainError", row.names = FALSE, quote = FALSE)

nnBenchmarkTrainError = read.csv("nnBenchmarkTrainError", header = TRUE)
options(scipen=999)
nnBenchmarkTrainError$variable = factor(nnBenchmarkTrainError$variable)

ggplot(data = nnBenchmarkTrainError, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 0.3)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") + 
  labs(colour = "batch size")

################################################################################ 
############################## Plot Test Errors ################################
################################################################################ 

nnBenchmarkTestError = as.data.frame(cbind(1:dim(resultsTest)[1], resultsTest))
colnames(nnBenchmarkTestError)[1] = "epoch"
nnBenchmarkTestError = melt(nnBenchmarkTestError, id = "epoch")

write.csv(nnBenchmarkTestError, file = "nnBenchmarkTestError", row.names = FALSE, quote = FALSE)

nnBenchmarkTestError = read.csv("nnBenchmarkTestError", header = TRUE)
options(scipen=999)
nnBenchmarkTestError$variable = factor(nnBenchmarkTestError$variable)

ggplot(data = nnBenchmarkTestError, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.3)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +  
  labs(colour = "batch size")

