

library(party)
library(ggplot2)

set.seed(1)
f = function(x) 0.5 * x^2 + x + sin(x)
x = runif(40, min = -3, max = 3)
y = f(x) + rnorm(40)
df = data.frame(x = x, y = y)
ggplot(df, aes(x, y)) + geom_point(size = 3) + stat_function(fun = f, color = "#FF9999", size = 2)

#############################################

ggplot() +
  stat_function(data = data.frame(x = c(-0.5, 4)), aes(x = x), fun = function (x) { 1 + 0.5 * x }) +
  geom_segment(mapping = aes(x = 2, y = 2, xend = 3, yend = 2), linetype = "dashed") +
  geom_segment(mapping = aes(x = 3, y = 2, xend = 3, yend = 2.5), linetype = "dashed", color = "red") +
  geom_segment(mapping = aes(x = -0.3, y = 1, xend = 0.3, yend = 1), linetype = "dashed", color = "blue") +
  geom_text(mapping = aes(x = 2.5, y = 2, label = "1 Unit"), vjust = 2) +
  geom_text(mapping = aes(x = 3, y = 2.25, label = "{theta == slope} == 0.5"), hjust = -0.05, parse = TRUE, color = "red") +
  geom_text(mapping = aes(x = 0, y = 1, label = "{theta[0] == intercept} == 1"), hjust = -0.25, parse = TRUE, color = "blue") +
  geom_vline(xintercept = 0, color = "gray", linetype = "dashed") +
  ylim(c(0, 3.5)) + xlim(c(-0.5, 4.3))

##############################################

set.seed(1)
x = 1:5
y = 0.2 * x + rnorm(length(x), sd = 2)
d = data.frame(x = x, y = y)
m = lm(y ~ x)
pl = ggplot(aes(x = x, y = y), data = d)
pl = pl + geom_abline(intercept = m$coefficients[1], slope = m$coefficients[2])
pl = pl + geom_rect(aes(ymin = y[3], ymax = y[3] + (m$fit[3] - y[3]), xmin = 3, xmax = 3 + abs(y[3] - m$fit[3])), color = "black", linetype = "dotted", fill = "transparent")
pl = pl + geom_rect(aes(ymin = y[4], ymax = y[4] + (m$fit[4] - y[4]), xmin = 4, xmax = 4 + abs(y[4] - m$fit[4])), color = "black", linetype = "dotted", fill = "transparent")
pl = pl + geom_segment(aes(x = 3, y = y[3], xend = 3, yend = m$fit[3]), color = "white")
pl = pl + geom_segment(aes(x = 4, y = y[4], xend = 4, yend = m$fit[4]), color = "white")
pl = pl + geom_segment(aes(x = 3, y = y[3], xend = 3, yend = m$fit[3]), linetype = "dotted", color = "red")
pl = pl + geom_segment(aes(x = 4, y = y[4], xend = 4, yend = m$fit[4]), linetype = "dotted", color = "red")
pl = pl + geom_point()
pl = pl + coord_fixed()
print(pl)

###############################################

source("code/plot_loss.R")

# generate some data, sample from line with gaussian errors
# make the leftmost obs an outlier
n = 10
x = sort(runif(n = n, min = 0, max = 10))
y = 3 * x + 1 + rnorm(n, sd = 5)
X = cbind(x0 = 1, x1 = x)
y[1] = 100

# fit l1/2 models on data without then with outlier data
b1 = optLoss(X[-1,], y[-1], loss = loss1)
b2 = optLoss(X[-1,], y[-1], loss = loss2)
b3 = optLoss(X, y, loss = loss1)
b4 = optLoss(X, y, loss = loss2)

# plot all 4 models
pl = plotIt(X, y, models = data.frame(
  intercept = c(b1[1], b2[1]),
  slope = c(b1[2], b2[2]),
  Loss = c("L2", "L1"),
  lty = rep(c("solid"), 2)
), remove_outlier = 1) + ylim(c(0, 100)) + ggtitle("L1 vs L2 Without Outlier")
print(pl)

#################################################

# plot all 4 models
pl = plotIt(X, y, models = data.frame(
  intercept = c(b3[1], b4[1]),
  slope = c(b3[2], b4[2]),
  Loss = c("L2", "L1"),
  lty = rep(c("solid"), 2)
), highlight_outlier = 1) + ylim(c(0, 100)) + ggtitle("L1 vs L2 With Outlier")
print(pl)

###############################################

library('e1071')
set.seed(1)
df2 = data.frame(
  x1 = c(rnorm(10, mean = 3), rnorm(10, mean = 5)),
  x2 = runif(10), class = rep(c("a", "b"), each = 10)
)
ggplot(df2, aes(x = x1, y = x2, shape = class, color = class)) +
  geom_point(size = 3) +
  geom_vline(xintercept = 4, linetype = "longdash")

################################################

library(mlr)
plotLearnerPrediction(makeLearner("classif.svm"), iris.task,
                      c("Petal.Length", "Petal.Width")) + ggtitle("")
################################################

n = 30
set.seed(1234L)
tar = factor(c(rep(1L, times = n), rep(2L, times = n)))
feat1 = c(rnorm(n, sd = 1.5), rnorm(n, mean = 2, sd = 1.5))
feat2 = c(rnorm(n, sd = 1.5), rnorm(n, mean = 2, sd = 1.5))
bin.data = data.frame(feat1, feat2, tar)
bin.tsk = makeClassifTask(data = bin.data, target = "tar")
plotLearnerPrediction("classif.logreg", bin.tsk)
###############################################

t = seq(-7, 7, 0.01)
resp = exp(t) / (exp(t) + 1L)
resp.dat = data.frame(t, resp)
q = ggplot(data = resp.dat, aes(t, resp)) + geom_line()
q = q + ylab(expression(tau(t)))
q
#################################################

library(mlbench)
library(BBmisc)
data = as.data.frame(mlbench.2dnormals(n = 200, cl = 3))
data$classes = mapValues(data$classes, "3", "1")
task = makeClassifTask(data = data, target = "classes")
learner = makeLearner("classif.ksvm")
plotLearnerPrediction(learner, task, kernel = "rbfdot", C = 1, sigma = 100, pointsize = 4)

###############################################
par(mar = c(2.1, 2.1, 0, 0))
x <- seq(0, 1, length.out = 20)
y1 <- c(1 - x[1:4], 1.5 - 4 * x[5:6], 0.5 - 0.8 * x[7:10], 0.15 - 0.1 * x[11:20])
X <- seq(0, 1, length.out = 1000)
Y1 <- predict(smooth.spline(x, y1, df = 10), X)$y
plot(X, Y1, type = "l", axes = FALSE, xlab = "Complexity", ylab = "Error")
mtext("Complexity", side = 1, line = 1)
mtext("Error", side = 2, line = 1)
Y2 <- 0.5 * X
lines(X, Y1 + Y2)
abline(v = 0.42, lty = 2)
text(0.4, 0.93, "Underfitting", pos = 2)
text(0.44, 0.93, "Overfitting", pos = 4)
arrows(0.4, 0.98, 0.2, 0.98, length = 0.1)
arrows(0.44, 0.98, 0.64, 0.98, length = 0.1)
box()
text(0.85, 0.13, "Apparent error")
text(0.85, 0.55, "Actual error")
#################################################

source("code/plot_loss_measures.R")

set.seed(31415)

x = 1:5
y = 2 + 0.5 * x + rnorm(length(x), 0, 1.5)
data = data.frame(x = x, y = y)
model = lm(y ~ x)

plotModQuadraticLoss(data = data, model = model, pt_idx = c(1,4))

################################################

plotModAbsoluteLoss(data, model = model, pt_idx = c(1,4))
##############################################
plotModAbsLogQuadLoss(data, model = model, pt_idx = c(1,4))
##############################################
source("code/plot_train_test.R")

.h = function(x) 0.5 + 0.4 * sin(2 * pi * x)
h = function(x) .h(x) + rnorm(length(x), mean = 0, sd = 0.05)

set.seed(1234)
x.all = seq(0, 1, length = 26L)
ind = seq(1, length(x.all), by = 2)

mydf = data.frame(x = x.all, y = h(x.all))


ggTrainTestPlot(data = mydf, truth.fun = .h, truth.min = 0, truth.max = 1, test.plot = FALSE,
                test.ind = ind)[["plot"]] + ylim(0, 1)

############################################

source("code/plot_train_test.R")
out = ggTrainTestPlot(data = mydf, truth.fun = .h, truth.min = 0, truth.max = 1, test.plot = FALSE,
                      test.ind = ind, degree = c(1, 5, 9))
out[["plot"]] + ylim(0, 1)

#############################################

source("code/plot_train_test.R")
ggTrainTestPlot(data = mydf, truth.fun = .h, truth.min = 0, truth.max = 1, test.plot = TRUE,
                test.ind = ind)[["plot"]] + ylim(0, 1)

#################################################
source("code/plot_train_test.R")
ggTrainTestPlot(data = mydf, truth.fun = .h, truth.min = 0, truth.max = 1, test.plot = TRUE,
                test.ind = ind, degree = c(1, 5, 9))[["plot"]] + ylim(0, 1)

#########################################################

source("code/plot_train_test.R")
degrees = 1:9

errors = ggTrainTestPlot(data = mydf, truth.fun = .h, truth.min = 0, truth.max = 1, test.plot = TRUE,
                         test.ind = ind, degree = degrees)[["train.test"]]

par(mar = c(4, 4, 1, 1))
#par(mar = c(4, 4, 0, 0) + 0.1)
plot(1, type = "n", xlim = c(1, 10), ylim = c(0, 0.1),
     ylab = "MSE", xlab = "degree of polynomial")
lines(degrees, sapply(errors, function(x) x["train"]), type = "b")
lines(degrees, sapply(errors, function(x) x["test"]), type = "b", col = "gray")

legend("topright", c("training error", "test error"), lty = 1L, col = c("black", "gray"))
text(3.75, 0.05, "High Bias,\nLow Variance", bg = "white")
arrows(4.75, 0.05, 2.75, 0.05, code = 2L, lty = 2L, length = 0.1)

text(6.5, 0.05, "Low Bias,\nHigh Variance", bg = "white")
arrows(7.5, 0.05, 5.5, 0.05, code = 1, lty = 2, length = 0.1)

#####################################################

library(reshape2)
bres = readRDS("data/cod_lm_rpart.rds")
bres = melt(bres, id.vars = c("task.id", "learner.id", "d"))

p = ggplot(data = bres[bres$d != 500, ], aes(x = d, y = value, colour = learner.id))
p = p + geom_line()
p = p + xlab("Number of noise variables") + ylab("Mean Squared Error") + labs(colour = "Learner")
p = p + ylim(c(0, 300))
p

########################################################

load("data/ozone_example.RData")

dfp =df_incdata[nobs == 50, ]

p = ggplot(data = dfp, aes(x = 0, y = value, colour = variable))
p = p + geom_boxplot() + labs(colour = " ")
p = p + scale_colour_discrete(labels = c("Train error", "Test error"))
p = p + xlab(" ") + ylab("Mean Squared Error")
p = p + ylim(c(0, 400)) + theme(axis.title.x=element_blank(),
                                axis.text.x=element_blank(),
                                axis.ticks.x=element_blank())
p
#####################################################

library(data.table)

dfp = setDT(df_incdata)[, .(mean.mse = mean(value)), by = c("nobs", "variable")]

p = ggplot(data = dfp, aes(x = nobs, y = mean.mse, colour = variable))
p = p + geom_line() + ylim(c(0, 500)) + labs(colour = " ")
p = p + scale_colour_discrete(labels = c("Train error", "Test error"))
p = p + xlab("Size of data set") + ylab("MSE")
p

#####################################################

load("data/ozone_example.RData")

p = ggplot(data = df_incfeatures, aes(x = type, y = mean.mse, colour = variable))
p = p + geom_line() + labs(colour = " ")
p = p + scale_colour_discrete(labels = c("Train error", "Test error"))
p = p + xlab("Number of features") + ylab("Mean Squared Error")
p = p + ylim(c(0, 200))
p = p + scale_x_continuous(breaks = 0:12)
p

###########################################

source("code/ridge_polynomial_reg.R")

set.seed(314259)
f = function (x) {
  return (5 + 2 * x + 10 * x^2 - 2 * x^3)
}

x = runif(40, -2, 5)
y = f(x) + rnorm(length(x), 0, 10)

x.true = seq(-2, 5, length.out = 400)
y.true = f(x.true)
df = data.frame(x = x.true, y = y.true)

lambda.vec = 0

plotRidge(x, y, lambda.vec, baseTrafo, degree = 10) +
  geom_line(data = df, aes(x = x, y = y), color = "red", size = 1) +
  xlab("x") + ylab("f(x)") + ggtitle("Predicting Using Linear Regression")

#################################################
source("code/ridge_polynomial_reg.R")

f = function (x) {
  return (5 + 2 * x + 10 * x^2 - 2 * x^3)
}

set.seed(314259)
x = runif(40, -2, 5)
y = f(x) + rnorm(length(x), 0, 10)

x.true = seq(-2, 5, length.out = 400)
y.true = f(x.true)
df = data.frame(x = x.true, y = y.true)

lambda.vec = c(0, 10, 100)

plotRidge(x, y, lambda.vec, baseTrafo, degree = 10) +
  geom_line(data = df, aes(x = x, y = y), color = "red", size = 1) +
  xlab("x") + ylab("f(x)") + ggtitle("Predicting Using Ridge Regression")

################################################

source("code/regularized_log_reg.R")

############################################










