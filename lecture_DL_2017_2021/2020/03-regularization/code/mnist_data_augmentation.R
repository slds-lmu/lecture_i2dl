require("mxnet")

setwd("C:/Users/Niklas/lectures/deeplearning/2017/02-regularization/mnist")
train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)

augment_vertical = function(data_matrix, pixels=6){
  storage_up = as.data.frame(matrix(0,nrow=nrow(data_matrix), ncol = 785))
  storage_up[,1:(784-pixels*28)] = data_matrix[,(28*pixels+1):784]
  storage_up[,785] = data_matrix[,785]
  colnames(storage_up)[785] = "Response"
  
  storage_down = as.data.frame(matrix(0,nrow=nrow(data_matrix), ncol = 785))
  storage_down[,(28*pixels+1):784] = data_matrix[,1:(784-pixels*28)]
  storage_down[,785] = data_matrix[,785]
  colnames(storage_down)[785] = "Response"
  
  storage = merge(storage_down, storage_up, by = colnames(storage_down), all = TRUE)
  colnames(data_matrix) = colnames(storage)
  storage = merge(storage, data_matrix, by = colnames(storage), all = TRUE)
  storage
}

train_jann = train
train_jann$y = train$label
train_jann = train_jann[, -1]

train_augmented = augment_vertical(train_jann, pixels = 6)

train_niki = data.frame(matrix(0, ncol = ncol(train_augmented), nrow = nrow(train_augmented)))
train_niki[, 1] = train_augmented[, 785]
train_niki[, 2:785] = train_augmented[, 1:784]

par(mfrow =c(4,4),mai = c(0,0,0,0)) 

# for(i in 1:4){
#   y <- as.matrix(augmented_mnist_train[i,1:784])
#   dim(y) <- c(28,28)
#   image(y[,nrow(y):1],axes = FALSE, col = gray(255:0 / 255))
#   text( 0.2, 0, augmented_mnist_train[i,785], cex = 3, col = 2, pos = c(3,4))
#   grid(28,28, lty = 3, col = "blue")
# }

train = data.matrix(train_niki)
test = data.matrix(test)
train.x = train[,-1]
train.y = train[,1]
train.x = t(train.x/255)
test_org = test
test = test[,-1]
test = t(test/255)
table(train.y)

#### model 1 with weight decay ####

data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 = mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 = mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name="sm")
devices = mx.cpu()
mx.set.seed(1337)
model  =  mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                      ctx=devices, 
                                      learning.rate = 0.1,
                                      wd = 0.001,
                                      momentum = 0.9,
                                      num.round = 10, 
                                      array.batch.size = 100,
                                      eval.metric = mx.metric.accuracy,
                                      initializer=mx.init.uniform(0.07),
                                      epoch.end.callback=mx.callback.log.train.metric(100))
preds = predict(model, test)
dim(preds)
pred.label= max.col(t(preds)) - 1
table(pred.label)
head(pred.label)
table(test_org[,1],pred.label)
sum(diag(table(test_org[,1],pred.label)))/1000
#### model 2 no weight decay ####
data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 = mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 = mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name="sm")
devices = mx.cpu()
mx.set.seed(1337)
model  =  mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                      ctx=devices, 
                                      learning.rate = 0.1,
                                      momentum = 0.9,
                                      num.round = 10,
                                      array.batch.size = 100,
                                      eval.metric = mx.metric.accuracy,
                                      initializer=mx.init.uniform(0.07),
                                      epoch.end.callback=mx.callback.log.train.metric(100))

preds = predict(model, test)
dim(preds)
pred.label= max.col(t(preds)) - 1
table(pred.label)
head(pred.label)
table(test_org[,1],pred.label)
sum(diag(table(test_org[,1],pred.label)))/1000
