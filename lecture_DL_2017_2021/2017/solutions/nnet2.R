library(R6) #Classes
library(mlbench) # Datasets
library(purrr) # Functional Programming Tools
library(BBmisc) # Helper functions

network = R6Class("network",
  public = list(
    ### Fields ###
    data = NULL,
    target = NULL,
    neurons = NULL,
    activation = NULL,
    output.act = NULL,
    learning.rate = NULL,
    weights = NULL,
    weights.acceleration = NULL,
    output.weights = NULL,
    output.weights.acceleration = NULL,
    loss = NULL,
    z.in = NULL,
    z.out = NULL,
    u.in = NULL,
    u.out = NULL,
    n = NULL,
    p = NULL,
    nu = NULL,
    lambda = NULL,
    ### CONSTRUCTOR ###
    initialize = function(data, target, neurons, activation, output.act, learning.rate, loss, nu = 0, lambda = 0) {
      data = cbind(intercept = 1, data) #add bias
      self$data = data
      self$target = target
      self$neurons = neurons
      self$activation  = activation
      self$output.act = output.act
      self$learning.rate = learning.rate
      self$n = nrow(data)
      self$p = ncol(data)
      n.weights = ncol(data) * neurons
      self$weights = matrix(rnorm(n.weights, 0.1), ncol = neurons)
      self$output.weights = matrix(rnorm(neurons + 1, 0.1), nrow = neurons + 1) #add bias
      self$loss = loss
      self$nu = nu
      self$lambda = lambda
      self$weights.acceleration = 0
      self$output.weights.acceleration = 0
    },
    ### METHODS ###
    predict = function(data) {
      self$z.in = data %*% self$weights
      self$z.out = apply(self$z.in, 1:2, self$activation$fun)
      self$u.in = cbind(1, self$z.out) %*% self$output.weights
      apply(self$u.in, 1, self$output.act$fun)
    },
    getBatches = function(batch.size) {
      inds = sample(seq_len(self$n))
      split(inds, ceiling(seq_along(inds) / batch.size))
    },
    train = function(epochs, batch.size = 4) {
      for(e in seq_len(epochs)) { # Start Epoch 1 here
        batches = self$getBatches(batch.size = batch.size)
        for (batch in batches) { # Iterate over Batches
          updates = lapply(batch, function(b) { # Calculate weight updates for each element in batch
            preds = self$predict(self$data[b,, drop = FALSE])
            y.out.deriv = self$loss$grad(self$target[b], preds)
            y.in.deriv = self$output.act$grad(preds)
            output.weight.update = y.out.deriv * y.in.deriv * c(1, self$z.out)
            z.out.deriv = apply(self$z.out, 1:2, self$activation$grad)
            weight.update = t(t((y.out.deriv * y.in.deriv * self$z.out * z.out.deriv)) %*% self$data[b,,drop=FALSE])
            list(output.weight.update = output.weight.update, weight.update = weight.update)
          })
          #Update Weights with average weight updates
          self$output.weights.acceleration =  self$nu * self$output.weights.acceleration - self$learning.rate * (self$lambda * self$output.weights + (reduce(extractSubList(updates, "output.weight.update", simplify = FALSE), `+`) / batch.size))
          self$output.weights = self$output.weights + self$output.weights.acceleration
          self$weights.acceleration = self$nu * self$weights.acceleration - self$learning.rate * (self$lambda * self$weights + (reduce(extractSubList(updates, "weight.update", simplify = FALSE), `+`) / batch.size))
          self$weights = self$weights + self$weights.acceleration
        }
        #Evaluate performance after Epoch
        p = self$predict(self$data)
        l = mean(self$loss$fun(self$target, p))
        acc = mean(round(p) == self$target)
        cat(sprintf("Epoch %i, with loss %f and accuracy %f\n", e, l, acc))
      }
    }
  )
)



relu = list(
  fun = function(x) max(c(0, x)),
  grad = function(x) as.numeric(x > 0)
)


sigmoid = list(
  fun = function(x) 1/(1 + exp(-x)),
  grad = function(x) exp(x) / (1 + exp(x)^2)
)


L2 = list(
  fun = function(y, yhat) (y - yhat)^2,
  grad = function(y, yhat) - (y - yhat)
)




data(Sonar, package = "mlbench")
data = as.matrix(Sonar[, -61])
target = as.numeric(Sonar[,61]) - 1

net = network$new(data = data, target = target, neurons = 5,
  activation = relu, output.act = sigmoid, learning.rate = 0.01, loss = L2, nu = 0.4, lambda = 0.01)
net$train(100, batch.size = 1)




xor = matrix(c(1,1,0,0,1,0,1,0), ncol=2)
target = c(0,1,1,0)

xor.net = network$new(data = xor, target = target, neurons = 3, activation = relu, output.act = sigmoid, learning.rate = 0.01, loss = L2)
xor.net$train(10000, batch.size = 2)

