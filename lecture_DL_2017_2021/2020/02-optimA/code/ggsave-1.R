
require("colorspace")
require("ggplot2")
require("grDevices")

foo = function(x, y) {
  -1 * sin(x) * dnorm(y, mean = pi / 2, sd = 0.8)
}

x = y = seq(0, pi, length = 50)
z = outer(x, y, foo)
p = c(list(list(.1, 3)), optim0(.1, 3, FUN = foo, maximum = FALSE))
sd_plot(phi = 25, theta = 20, xlab = expression(paste(theta[1])), ylab = expression(paste(theta[2])), labels = FALSE)