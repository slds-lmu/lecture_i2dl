require("ggplot2")
require("smoof")
# function used to show physical intention of momention
p <- ggplot(data = data.frame(x = 0), mapping = aes(x = x))
fun.1 <- function(x) 5*((2*(x-4)) * (2*(x-4)) / 100.0 + 0.3 * sin(2*(x-4)) * sin(2*(x-4)))
p + stat_function(fun = fun.1) + 
  scale_y_continuous(labels = function (x) floor(x), limits = c(0, 5), breaks = c(0, 5)) +
  scale_x_continuous(labels = function (x) floor(x), limits = c(0, 3), breaks = c(0:3))
# function used to show how momentum learns when dealing with ravins
fn = makeSphereFunction(2)
autoplot(fn, show.optimum = TRUE, title = FALSE, contour = TRUE)  


