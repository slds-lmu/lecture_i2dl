perceptron = function(x, y, eta, epochs, viz = FALSE) {

    theta = numeric(ncol(x) + 1)

    for (i in seq_len(epochs)) {
        for (n in seq_along(y)) {
            z = sum(theta[-1] * x[n, ]) + theta[1]
            ypred = sign(z)
            delta = eta * (y[n] - ypred) * c(theta0 = 1, unlist(x[n, ]))
            theta = theta + delta

            if(viz) {
             plot(iris[,1], iris[,2], col = iris[,3], xlim = c(0, 10), ylim = c(0, 7))
             abline(theta[1] / theta[2], -1 * theta[2] / theta[3])
             Sys.sleep(0.1)
            }
        }
    }
    return(theta)
}

iris = iris[1:100, c(1, 3, 5)]
colnames(iris) = c("sepal", "petal", "species")
head(iris)

x = iris[,-3]
y = ifelse(iris$species == "setosa", 1, -1)
pars = perceptron(x, y, 1, 10, viz = TRUE)

plot(iris[,1], iris[,2], col = iris[,3])
 abline(pars[1] / pars[2], -1 * pars[2] / pars[3])
