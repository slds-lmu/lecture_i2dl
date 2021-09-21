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
################################# autoencoder ##################################
################################################################################
data = mx.symbol.Variable("data")
# label = mx.symbol.Variable("label") # nötig bei custom loss function

encoder = mx.symbol.FullyConnected(data, num_hidden = 64) # mit identity activation beste Ergebnisse
# dropout = mx.symbol.Dropout(encoder, p = 0.2)
# activation1 = mx.symbol.Activation(encoder, act_type = "relu")
decoder = mx.symbol.FullyConnected(encoder, num_hidden = 784)
activation2 = mx.symbol.Activation(decoder, act_type = "sigmoid")
output = mx.symbol.LinearRegressionOutput(activation2)
# output = mx.symbol.MakeLoss(mx.symbol.square(activation2 - label))

mx.set.seed(1337)
logger = mx.metric.logger$new()
devices = mx.cpu()

model = mx.model.FeedForward.create(output,
  X = train.x, y = train.x,
  ctx = mx.cpu(),
  num.round = 50, # lernt selbst nach 1000 Epochen noch weiter
  array.batch.size = 32, # je größer desto schlechter
  #optimizer = "adadelta", # sehr schlecht
  #optimizer = "sgd", learning.rate = 0.03, momentum = 0.9, wd = 0.001, # sehr schlecht
  #optimizer = "rmsprop", # sehr schlecht
  #optimizer = "adagrad", # gut
  optimizer = "adam", # am besten
  initializer = mx.init.uniform(0.01), # je größer scale, desto länger bis konvergenz
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
# how many mnist digits to plot
plots = 4

# sample random mnist digits from the test set to plot
# visualize = sample(dim(test.x)[2], plots) # comment this line if u want same digits for different architectures

# plot layout
par(mfcol = c(2, plots), cex = 0.1)

for(i in 1:plots){
  
  # truth
  truth = test.x[1:784, visualize[i]]
  truth_mat = matrix(truth, nrow = 28, ncol = 28, byrow = TRUE)
  # since the image function is very stupid and rotates the mnist digits by 90 degree,
  # we have to reverse the data matrix and plot the transpose to obtain the actual image 
  truth_mat = apply(truth_mat, 2 , rev)
  image(t(truth_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
  
  # autoencoder prediction
  autoencoder = pred[1:784, visualize[i]]
  autoencoder_mat = matrix(autoencoder, nrow = 28, ncol = 28, byrow = TRUE)
  autoencoder_mat = apply(autoencoder_mat, 2 , rev)
  image(t(autoencoder_mat), axes = FALSE, col = grey(seq(from = 0, to = 1, length = 255)))
}

# dev.off()