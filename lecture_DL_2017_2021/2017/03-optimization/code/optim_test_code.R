source("../code/functions.R")
require("smoof")
require("colorspace")
require("ggplot2")
# this file contains functions which might be used to show how optim algorithms behave
###########################################################################
############################### slides ####################################
###########################################################################
foo = function(x, y) {
  -1 * sin(x) * dnorm(y, mean = pi / 2, sd = 0.8)
}
x = y = seq(0, pi, length = 50)
z = outer(x, y, foo)
p = c(list(list(.1, 3)), optim0(0, 3, FUN = foo, maximum = FALSE))
sd_plot(phi = 25, theta = 20, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut #####################################
###########################################################################
foo = function(x, y) {
  0.6 + (((sin(1 - 16/15*x))^2 - 1/50*sin(4 - 64/15*x) - sin(1 - 16/15*x))+
           ((sin(1 - 16/15*y))^2-1/50*sin(4 - 64/15*y) - sin(1 - 16/15*y)))
}
x = y = seq(-1.7, 3, length = 30)
z = outer(x, y, foo)
p = c(list(list(2.7, 2.7)), optim0(0.1, 2, FUN = foo, maximum = FALSE, maxit = 100))
sd_plot(phi = 30, theta = 320, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut2 ####################################
###########################################################################
foo = function(x, y) {
  0.6 + (((sin(1 - 16/15*x))^2 - 1/50*sin(4 - 64/15*x) - sin(1 - 16/15*x))+
           ((sin(1 - 16/15*y))^2-1/50*sin(4 - 64/15*y) - sin(1 - 16/15*y)))
}
x = y = seq(-1.4, 1.4, length = 30)
z = outer(x, y, foo)
p = c(list(list(0.5, 1.4)), optim0(.1, .2, FUN = foo, maximum = FALSE, maxit = 100))
sd_plot(phi = 30, theta = -20, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut3 ####################################
###########################################################################
foo = function(x, y) {
  0.6 + (((sin(1 - 16/15*x))^2 - 1/50*sin(4 - 64/15*x) - sin(1 - 16/15*x))+
           ((sin(1 - 16/15*y))^2-1/50*sin(4 - 64/15*y) - sin(1 - 16/15*y)))
}
x = y = seq(-1.4, 1.4, length = 30)
z = outer(x, y, foo)
p = c(list(list(0.5, 1.4)), optim0(0.4, .3, FUN = foo, maximum = FALSE, maxit = 50))
sd_plot(phi = 30, theta = -20, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut4 ####################################
###########################################################################
foo = function(x, y) {
  (x - y)^2 + exp((1 - sin(x))^2) * cos(y) + exp((1 - cos(y))^2) * sin(x)
}
x = y = seq(-10, 10, length = 50)
z = outer(x, y, foo)
p = c(list(list(-100, 100)))
sd_plot(phi = 40, theta = 185, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut5 ####################################
###########################################################################
foo = function(x, y){
  x^2 - y^2
}
x = y = seq(-2, 2, length = 30)
z = outer(x, y, foo)
p = c(list(list(1, 0)), optim0(.2, 0, FUN = foo, maximum = FALSE, maxit = 10))
#p = c(list(list(-100, 100)))
sd_plot(phi = 40, theta = 190, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut6 ####################################
###########################################################################
foo = function(x, y){
  100 * (y - x^3)^2 + (1 - x)^2
}
x = y = seq(0, 5, length = 30)
z = outer(x, y, foo)
p = c(list(list(1, 0)), optim0(.2, 0, FUN = foo, maximum = FALSE, maxit = 10))
#p = c(list(list(-100, 100)))
sd_plot(phi = 40, theta = 190, xlab = "x1", ylab = "x2")
###########################################################################
################################# gut6 ####################################
###########################################################################
fn = makeBartelsConnFunction()
autoplot(fn, show.optimum = TRUE, title = FALSE, contour = TRUE)
# many many optima
fn = makeEggCrateFunction()
autoplot(fn, show.optimum = TRUE, title = FALSE, contour = TRUE)
# super many minima
fn = makePeriodicFunction()
autoplot(fn, show.optimum = TRUE, title = FALSE, contour = TRUE)  
# 
fn = makeSphereFunction(2)
autoplot(fn, show.optimum = TRUE, title = FALSE, contour = TRUE)  
#



  
