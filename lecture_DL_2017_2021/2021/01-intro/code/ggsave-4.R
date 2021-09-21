

library(ggplot2)
library(reshape2)

######################################## ReLU
relu <- function(x) sapply(x, function(z) max(0,z))
x <- seq(from=-5, to=5, by=0.1)
fits <- data.frame(x=x, relu = relu(x))
long <- melt(fits, id.vars="x")
ggplot(data=long, aes(x=x, y=value, group=variable, colour=variable))+
  geom_line(size=1.5) + scale_y_continuous(name = NULL) + 
  scale_x_continuous(name = NULL) + theme(legend.position="none")

######################################### tanh

relut <- function(x) sapply(x, function(z) tanh(z))
x <- seq(from=-3, to=3, by=0.1)
fits <- data.frame(x=x, relut = relut(x))
long <- melt(fits, id.vars="x")
ggplot(data=long, aes(x=x, y=value, group=variable, colour=variable))+
  geom_line(size=1.2) + scale_y_continuous(name = NULL) + 
  scale_x_continuous(name = NULL) + theme(legend.position="none")

######################################### Sigmoid

logfun = function(v, s) {
  1 / (1 + exp(-v))
}
x = seq(-10, 10, 0.1)
stretch = c(1)
y = sapply(stretch, function(s) {
  sapply(x, logfun, s = s)
})
df = data.frame(y = as.vector(y), x = rep(x, length(stretch)),
                s = as.factor(rep(stretch, each = length(x))))

logfun.q = ggplot(df, aes(x = x, y = y, color = s)) + geom_line(size=1)
logfun.q = logfun.q + scale_y_continuous(name = NULL)
logfun.q = logfun.q + scale_x_continuous(name = NULL)
logfun.q = logfun.q + theme(legend.position="none")
logfun.q = logfun.q + theme(axis.title = element_text(size = 14L, face = "bold"),
                            plot.margin = unit(c(0, 0, 0, 0), "cm"))
logfun.q

############################################# reg

library("mlr")
set.seed(1234L)
n = 50L
x = sort(10 * runif(n))
y = sin(x) + 0.2 * rnorm(x)
df = data.frame(x = x, y = y)
tsk = makeRegrTask("sine function example", data = df, target = "y")
plotLearnerPrediction("regr.nnet", tsk, size = 1L, maxit = 1000)

############################################# class

library("mlr")
library("mlbench")
set.seed(1234L)
spirals = mlbench.spirals(500,1.5,0.05)
spirals = data.frame(cbind(spirals$x, spirals$classes))
colnames(spirals) = c("x1","x2","class")
spirals$class = as.factor(spirals$class)
task = makeClassifTask(data = spirals, target = "class")
lrn = makeLearner("classif.nnet")
plotLearnerPrediction("classif.nnet", task, size = 1L, maxit = 500)

#############################################









