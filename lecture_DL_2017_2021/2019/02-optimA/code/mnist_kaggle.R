################################################################################ 
################################# preparations #################################
################################################################################ 
require("mxnet")
require("ggplot2")
require("reshape2")

train = read.csv("code/train.csv", header = TRUE)
test = read.csv("code/test.csv", header = TRUE)

train = data.matrix(train)
test = data.matrix(test)
train.x = train[, -1]
train.y = train[, 1]
train.x = t(train.x/255)
test = t(test/255)

################################################################################ 
############################ model architecture ################################
################################################################################
data = mx.symbol.Variable(name = "data")

# layer + activation functions:
layer1 = mx.symbol.FullyConnected(data = data, name = "layer 1", num_hidden = 10)
activation1 = mx.symbol.Activation(data = layer1, name = "function 1", act_type = "relu")
layer2 = mx.symbol.FullyConnected(data = activation1, name = "layer 2", num_hidden = 10)
activation2 = mx.symbol.Activation(data = layer2, name = "function 2", act_type = "relu")
layer3 = mx.symbol.FullyConnected(data = activation2, name = "layer 3", num_hidden = 10)

# in mxnet, the loss function is defined by the output layer:
output = mx.symbol.SoftmaxOutput(data = layer3, name = "function")

devices = mx.cpu()
mx.set.seed(1337)

model = mx.model.FeedForward.create(
  symbol = output, 
  X = train.x, y = train.y,
  ctx = devices, 
  num.round = 10, 
  array.batch.size = 128,
  learning.rate = 0.05, 
  momentum = 0.9,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback = mx.callback.log.train.metric(100)
  )

################################################################################ 
################################ visualization #################################
################################################################################
graph.viz(model$symbol)

################################################################################ 
################################ prediction ####################################
################################################################################
preds = predict(model, test)
# this yields us predicted probabilities for all 10 classes
dim(preds)

# we choose the maximum to obtain quantities for each class
pred.label = max.col(t(preds)) - 1
table(pred.label)

################################################################################ 
############################# kaggle submission ################################
################################################################################
submission = data.frame(ImageId = 1:ncol(test), Label = pred.label)

write.csv(submission, file = 'submission.csv', 
  row.names = FALSE, quote = FALSE)
