################################################################################
################################# preparations #################################
################################################################################
library(mxnet)
library(readr)
library(BBmisc)
library(imager)

train = read_csv("~/Downloads/train.csv")
test = read_csv("~/Downloads/test.csv")


imPreproc = function(v) {
  as.cimg(array(v, c(32, 32, 1, 3))) %>%
    imrotate(angle = runif(1, -90, 90), boundary = 1) %>%
    imshift(delta_x = runif(1, -10, 10),
      delta_y = runif(1, -10, 10), boundary = 1) %>%
    resize(size_x = 32, size_y = 32, size_c = 3)
}



plotImg = function(x) plot(as.cimg(array(x, c(32, 32, 1, 3))))



################################################################################
############################## image processing ################################
################################################################################

train[, -ncol(train)] = train[, -ncol(train)] / 255
train = data.matrix(train)

c = ncol(train)
n = nrow(train)

val.inds = sample.int(n, 3334)
train.inds = setdiff(seq_len(n), val.inds)

val = train[val.inds, ]
train = train[train.inds, ]


train2 = apply(train, 1, function(x) {
  c(as.numeric(imPreproc(x[-c])), x[c])
})

train = rbind(train, t(train2))
train = train[sample(seq_len(nrow(train))), ]

train.y = train[,c]
val.y = val[,c]

train = train[,-c]
val = val[,-c]

test.ids = test$id
test = data.matrix(test[,-ncol(test)]) / 255


data.shape = c(32, 32, 3)

train = array(aperm(train), dim = c(data.shape, nrow(train)))
val = array(aperm(val), dim = c(data.shape, nrow(val)))
test = array(aperm(test), dim = c(data.shape, nrow(test)))


eval.data = list(label = val.y, data = val)
################################################################################
############################ model architecture ################################
################################################################################
data = mx.symbol.Variable('data')
network = mx.symbol.Dropout(data, p = 0.2)
network = mx.symbol.Convolution(data = network, kernel = c(3, 3), num_filter = 32)
network = mx.symbol.Activation(data = network, act_type = "relu")
network = mx.symbol.Pooling(data = network, pool_type = "max", kernel = c(2, 2), stride = c(1, 1))
network = mx.symbol.Dropout(network, p = 0.2)
network = mx.symbol.Convolution(data = network, kernel = c(3, 3), num_filter = 32)
network = mx.symbol.Activation(data = network, act_type = "relu")
network = mx.symbol.Pooling(data= network, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
network = mx.symbol.Dropout(network, p = 0.2)
network = mx.symbol.Flatten(data = network)
network = mx.symbol.FullyConnected(data = network, num_hidden = 200)
network = mx.symbol.Activation(data = network, act_type = "relu")
network = mx.symbol.FullyConnected(data = network, num_hidden = 10)
network = mx.symbol.SoftmaxOutput(data = network)


devices = mx.gpu()
mx.set.seed(1337)

model = mx.model.FeedForward.create(
  symbol = network,
  X = train, y = train.y,
  ctx = devices,
  num.round = 100,
  array.batch.size = 32,
  learning.rate = 0.005,
  momentum = 0.9,
  wd = 0.01,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.Xavier(),
  eval.data = eval.data,
  epoch.end.callback = mxnet::mx.callback.early.stop(bad.steps = 10, maximize = TRUE)
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
submission = data.frame(class = pred.label, id = test.ids)

write.csv(submission, file = 'submission.csv',
  row.names = FALSE, quote = FALSE)
