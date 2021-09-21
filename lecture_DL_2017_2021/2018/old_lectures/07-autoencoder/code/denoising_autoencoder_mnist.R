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
noise_level = 0.55 # 0.1: almost no noise, 1: alot (!) of noise
train.x_noised = train.x + runif(dim(train.x)[1] * dim(train.x)[2], 
  min = - noise_level, max = noise_level)
train.x_noised[train.x_noised < 0] = 0
train.x_noised[train.x_noised > 1] = 1

test.x_noised = test.x + runif(dim(test.x)[1] * dim(test.x)[2], 
  min = - noise_level, max = noise_level)
test.x_noised[test.x_noised < 0] = 0
test.x_noised[test.x_noised > 1] = 1

plots = 4
# visualize_noise = sample(dim(train.x)[2], plots) # for slides (2825 4486 3119 1470) were sampled
par(mfcol = c(2, plots), cex = 0.1)

for(i in 1:plots){
  
  truth = train.x[1:784, visualize_noise[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
  noise = train.x_noised[1:784, visualize_noise[i]]
  noise_mat = matrix(noise, nrow = 28, ncol = 28, byrow = TRUE)
  noise_mat = apply(noise_mat, 2 , rev)
  image(t(noise_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
}

################################################################################ 
################################# autoencoder ##################################
################################################################################
data = mx.symbol.Variable("data")

encoder = mx.symbol.FullyConnected(data, num_hidden = 8)
decoder = mx.symbol.FullyConnected(encoder, num_hidden = 784)
activation2 = mx.symbol.Activation(decoder, act_type = "sigmoid")
output = mx.symbol.LinearRegressionOutput(activation2)

mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()

model = mx.model.FeedForward.create(output,
  X = train.x_noised, y = train.x,
  ctx = mx.cpu(),
  num.round = 50,
  array.batch.size = 64,
  optimizer = "adam",
  initializer = mx.init.uniform(0.01),
  eval.metric = mx.metric.mse,
  epoch.end.callback = mx.callback.log.train.metric(100, logger),
  array.layout = "colmajor")

################################################################################ 
################################ prediction ####################################
################################################################################
pred = predict(model, test.x_noised)
dim(pred)

################################################################################ 
############################### visualization ##################################
################################################################################
plots = 4
# visualize_pred = sample(dim(test.x)[2], plots) # for slides (104 51 680 862) were sampled
par(mfcol = c(3, plots), cex = 0.02)

for(i in 1:plots){
  
  # denoised truth
  truth = test.x[1:784, visualize_pred[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))  
    
  # noised data
  noise = test.x_noised[1:784, visualize_pred[i]]
  noise_mat = matrix(noise, nrow = 28, ncol = 28, byrow = TRUE)
  noise_mat = apply(noise_mat, 2 , rev)
  image(t(noise_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255))) 
  
  # autoencoder prediction
  autoencoder = pred[1:784, visualize_pred[i]]
  autoencoder_mat = matrix(autoencoder, nrow = 28, ncol = 28, byrow = TRUE)
  autoencoder_mat = apply(autoencoder_mat, 2 , rev)
  image(t(autoencoder_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
}

# dev.off()