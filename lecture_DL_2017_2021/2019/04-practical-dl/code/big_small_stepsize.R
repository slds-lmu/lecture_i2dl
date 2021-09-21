devtools::install_github("PhilippScheller/visualDescent", force = TRUE)

library(visualDescent)
library(smoof)

f = makeSingleObjectiveFunction(name = "Paraboloid", 
	fn = function(x) 0.1 * x[1]^2 + 2 * x[2]^2, 
	par.set = makeNumericParamSet(
		"x", len = 2, lower = -10, upper = 10),
	global.opt.params = c(0, 0)
	)

gd1 = gradDescent(f = f, x0 = c(- 10, - 10), step.size = 0.48, max.iter = 100)$results
gd2 = gradDescent(f = f, x0 = c(- 10, - 10), step.size = 0.01, max.iter = 100)$results

results = list(gd1 = gd1, gd2 = gd2)
results = results[c("gd1", "gd2")]

p = plot2d(f = f, trueOpt = as.numeric(getGlobalOptimum(f)$param),
        xmat = results, algoName = list("big step size", "small step size"))
p

ggsave("plots/big_small_stepsize.png", p, width = 6, height = 3)