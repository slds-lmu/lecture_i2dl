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

train.x = train[ ,-1]
train.y = train[ ,1]
train.x = t(train.x/255)
dim(train.x)
rm(train)

test.x = test[ ,-1]
test.y = test[, 1]
test.x = t(test.x/255)
dim(test.x)
rm(test)
################################################################################ 
################################## noise data ##################################
################################################################################
noise.x = train.x + runif(dim(train.x)[1] * dim(train.x)[2], min = -0.5, max = 0.5)
noise.x[noise.x < 0] = 0
noise.x[noise.x > 1] = 1

plots = 4
visualize_noise = sample(dim(train.x)[2], plots)
par(mfcol = c(2, plots), cex = 0.1)

for(i in 1:plots){
  
  truth = train.x[1:784, visualize_noise[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
  noise = noise.x[1:784, visualize_noise[i]]
  noise_mat = matrix(noise, nrow = 28, ncol = 28, byrow = TRUE)
  noise_mat = apply(noise_mat, 2 , rev)
  image(t(noise_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
}

################################################################################ 
################################# autoencoder ##################################
################################################################################
data = mx.symbol.Variable("data")

encoder1 = mx.symbol.FullyConnected(data, num_hidden = 128)
activation1 = mx.symbol.Activation(encoder1, act_type = "relu")
encoder2 = mx.symbol.FullyConnected(activation1, num_hidden = 64)
activation2 = mx.symbol.Activation(encoder2, act_type = "relu")
encoder3 = mx.symbol.FullyConnected(activation2, num_hidden = 32)
activation3 = mx.symbol.Activation(encoder3, act_type = "relu")

decoder1 = mx.symbol.FullyConnected(activation3, num_hidden = 64)
activation4 = mx.symbol.Activation(decoder1, act_type = "relu")
decoder2 = mx.symbol.FullyConnected(activation4, num_hidden = 128)
activation5 = mx.symbol.Activation(decoder2, act_type = "relu")
decoder3 = mx.symbol.FullyConnected(activation5, num_hidden = 784)
activation6 = mx.symbol.Activation(decoder3, act_type = "sigmoid")
output = mx.symbol.LinearRegressionOutput(activation6)

mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()

model = mx.model.FeedForward.create(output,
  X = noise.x, y = train.x,
  ctx = mx.cpu(),
  num.round = 100,
  array.batch.size = 64,
  optimizer = "adam",
  initializer = mx.init.uniform(0.01),
  eval.metric = mx.metric.mse,
  epoch.end.callback = mx.callback.log.train.metric(100, logger),
  array.layout = "colmajor")

################################################################################ 
################################ prediction ####################################
################################################################################
pred = predict(model, test.x)
dim(pred)

################################################################################ 
############################### visualization ##################################
################################################################################
plots = 4
visualize_pred = sample(dim(test.x)[2], plots)
par(mfcol = c(2, plots), cex = 0.1)

for(i in 1:plots){
  
  # truth
  truth = test.x[1:784, visualize_pred[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
  # autoencoder prediction
  autoencoder = pred[1:784, visualize_pred[i]]
  autoencoder_mat = matrix(autoencoder, nrow = 28, ncol = 28, byrow = TRUE)
  autoencoder_mat = apply(autoencoder_mat, 2 , rev)
  image(t(autoencoder_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
}

# dev.off()