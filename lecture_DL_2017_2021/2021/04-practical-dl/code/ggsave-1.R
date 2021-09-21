
library(smoof)
library(ggplot2)
library(plot3D)
foo = function(x, y) {
  0.5 * x^2 + 4 * y^2
}
x = y = seq(-10, 10, length = 50)
z = outer(x, y, foo)
p = c(list(list(-100, 100)))
sd_plot(phi = 45, theta = 60, xlab = "x1", ylab = "x2")

##############################################

library(smoof)
library(ggplot2)
library(plot3D)
foo = function(x, y) {
  (x - y)^2 + exp((1 - sin(x))^2) * cos(y) + exp((1 - cos(y))^2) * sin(x)
}
x = y = seq(-10, 10, length = 50)
z = outer(x, y, foo)
p = c(list(list(-100, 100)))
sd_plot(phi = 40, theta = 185, xlab = "x1", ylab = "x2")

############################################

