devtools::install_github("PhilippScheller/visualDescent", force = TRUE)

library(visualDescent)
library(smoof)

f = makeStyblinkskiTangFunction()

gd = gradDescent(f = f, x0 = c(5, - 5), step.size = 0.002, max.iter = 200)$results
momentum = gradDescentMomentum(f = f, x0 = c(5, - 5), step.size = 0.002, max.iter = 200, phi = 0.9)$results

results = list(gd = gd, momentum = momentum)
results = results[c("gd", "momentum")]

p = plot2d(f = f, trueOpt = as.numeric(getGlobalOptimum(f)$param),
        xmat = results, algoName = list("Gradient Descent", paste("Momentum", expression(eta), "= 0.5")))
p

ggsave("plots/momentum2.png", p, width = 7, height = 4)