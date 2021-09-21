################################################################################ 
################################# preparations #################################
################################################################################ 
require("mxnet")
require("ggplot2")
require("reshape2")
setwd("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization")
# train data: 5000 rows with 28*28 = 784 + 1 (label) columns
train = read.csv("train_short.csv", header = TRUE)
# test data: 1000 rows with 28*28 = 784 + 1 (label) columns
test = read.csv("test_short.csv", header = TRUE)
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
epochs = 500
batchSize = 100
learningRate = 0.03
dropoutInput = 0.4
dropoutLayer = 0.2
neuronsLayer1 = 512
neuronsLayer2 = 512
neuronsLayer3 = 512
################################################################################ 
#################### architecture 1 without dropout ############################
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
################################################################################ 
################################### training ###################################
################################################################################ 
devices = mx.cpu()
mx.set.seed(1)
logger1 = mx.metric.logger$new()
model1 = mx.model.FeedForward.create(softmax, 
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  learning.rate = learningRate,
  momentum = 0.9,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger1))

# compute errors instead of accuracy    
model1Error = lapply(list("train" = logger1$train, "test" = logger1$eval), function(x) 1-x)
################################################################################ 
#################################### plot 1 ####################################
################################################################################ 
error1 = as.data.frame(cbind(model1Error$train, model1Error$test, 1:length(model1Error$train)))
colnames(error1) = c("training error", "test error", "epoch")
error1 = melt(error1, id = "epoch")

write.csv(error1, file = "error1Data", row.names = FALSE, quote = FALSE)
error1plot = read.csv("error1Data", header = TRUE)

ggplot(data = error1plot, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "error rate", limits = c(0, 0.55)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))
################################################################################ 
###################### architecture 2 with dropout #############################
################################################################################ 
data = mx.symbol.Variable("data")
drop0 = mx.symbol.Dropout(data = data, p = dropoutInput)
fc1 = mx.symbol.FullyConnected(drop0, name = "fc1", num_hidden = neuronsLayer1)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
drop1 = mx.symbol.Dropout(data = act1, p = dropoutLayer)
fc2 = mx.symbol.FullyConnected(drop1, name = "fc2", num_hidden = neuronsLayer2)
act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
drop2 = mx.symbol.Dropout(data = act2, p = dropoutLayer)
fc3 = mx.symbol.FullyConnected(drop2, name = "fc3", num_hidden = neuronsLayer3)
act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
drop3 = mx.symbol.Dropout(data = act3, p = dropoutLayer)
fc4 = mx.symbol.FullyConnected(drop3, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
################################################################################ 
################################### training ###################################
################################################################################ 
devices = mx.cpu()
mx.set.seed(1)
logger2 = mx.metric.logger$new()
model2 = mx.model.FeedForward.create(softmax, 
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
    
model2Error = lapply(list("train" = logger2$train, "test" = logger2$eval), function(x) 1-x)
################################################################################ 
#################################### plot 2 ####################################
################################################################################ 
error2 = as.data.frame(cbind(model2Error$train, model2Error$test,
  1:length(model2Error$train)))
colnames(error2) = c("training error dropout", "test error dropout", "epoch")
error2 = melt(error2, id = "epoch")

write.csv(error2, file = "error2Data", row.names = FALSE, quote = FALSE)
error2plot = read.csv("error2Data", header = TRUE)

ggplot(data = error2plot, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "error rate", limits = c(0, 0.55)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))
################################################################################ 
################ architecture 3 with dropout and weight decay ##################
################################################################################ 
data = mx.symbol.Variable("data")
drop0 = mx.symbol.Dropout(data = data, p = dropoutInput)
fc1 = mx.symbol.FullyConnected(drop0, name = "fc1", num_hidden = neuronsLayer1)
act1 = mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
drop1 = mx.symbol.Dropout(data = act1, p = dropoutLayer)
fc2 = mx.symbol.FullyConnected(drop1, name = "fc2", num_hidden = neuronsLayer2)
act2 = mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
drop2 = mx.symbol.Dropout(data = act2, p = dropoutLayer)
fc3 = mx.symbol.FullyConnected(drop2, name = "fc3", num_hidden = neuronsLayer3)
act3 = mx.symbol.Activation(fc3, name = "relu3", act_type = "relu")
drop3 = mx.symbol.Dropout(data = act3, p = dropoutLayer)
fc4 = mx.symbol.FullyConnected(drop3, name = "fc4", num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")
################################################################################ 
################################### training ###################################
################################################################################ 
devices = mx.cpu()
mx.set.seed(1)
logger3 = mx.metric.logger$new()
model3 = mx.model.FeedForward.create(softmax, 
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  learning.rate = learningRate,
  momentum = 0.9,
  wd = 0.001,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger3))
    
model3Error = lapply(list("train" = logger3$train, "test" = logger3$eval), function(x) 1-x)
################################################################################ 
#################################### plot 3 ####################################
################################################################################ 
error3 = as.data.frame(cbind(model3Error$train, model3Error$test,
  1:length(model3Error$train)))
colnames(error3) = c("training error dropout + wd", "test error dropout + wd", "epoch")
error3 = melt(error3, id = "epoch")
write.csv(error3, file = "error3Data", row.names = FALSE, quote = FALSE)
error3plot = read.csv("error3Data", header = TRUE)

ggplot(data = error3plot, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "error rate", limits = c(0, 0.55)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))
test = t(test/255)
################################################################################ 
##################### architecture 4 only weight decay #########################
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
################################################################################ 
################################### training ###################################
################################################################################ 
devices = mx.cpu()
mx.set.seed(1)
logger4 = mx.metric.logger$new()
model4 = mx.model.FeedForward.create(softmax, 
  X = train.x, y = train.y,
  eval.data = list(data = test.x, label = test.y),
  ctx = devices,
  learning.rate = learningRate,
  momentum = 0.9,
  wd = 0.001,
  num.round = epochs,
  array.batch.size = batchSize,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(5, logger4))

# compute errors instead of accuracy    
model4Error = lapply(list("train" = logger4$train, "test" = logger4$eval), function(x) 1-x)
################################################################################ 
#################################### plot 1 ####################################
################################################################################ 
error4 = as.data.frame(cbind(model4Error$train, model4Error$test, 1:length(model4Error$train)))
colnames(error4) = c("training error", "test error", "epoch")
error4 = melt(error4, id = "epoch")

write.csv(error4, file = "error4Data", row.names = FALSE, quote = FALSE)
error4plot = read.csv("error4Data", header = TRUE)

ggplot(data = error4plot, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "error rate", limits = c(0, 0.55)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))
################################################################################ 
#################################### plot 5 ####################################
################################################################################
errors = as.data.frame(cbind(model1Error$test, 
  model2Error$test, 
  model3Error$test,
  model4Error$test,
  1:length(model3Error$train)))
colnames(errors) = c("unregularized", 
  "dropout", 
  "dropout + weight decay",
  "weight decay",
  "epoch")
errors = melt(errors, id = "epoch")

write.csv(errors, file = "errorsTotal", row.names = FALSE, quote = FALSE)
errorsTotal = read.csv("errorsTotal", header = TRUE)

ggplot(data = errorsTotal, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.2)) + 
  scale_x_continuous(labels = function (x) floor(x), 
    name = "epochs") +
  theme(legend.position = c(.78, .84), 
        legend.background = element_rect(color = "transparent", fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size = 14))




