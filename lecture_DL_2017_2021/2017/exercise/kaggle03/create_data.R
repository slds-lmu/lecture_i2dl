library(readr)
library(OpenML)
library(imager)

d = getOMLDataSet(40926)


data = d$data

n = nrow(data)
set.seed(1223)
data = data[sample.int(n),]

test.inds = sample.int(n, size = floor(n / 3))

train = data[-test.inds,]
test = data[test.inds,]
class = test$class
test$class = NULL
test$id = test.inds
write_csv(train, path = "train.csv")
write_csv(test, path = "test.csv")
write_csv(data.frame(class = class, id = test.inds), path = "solution.csv")
write_csv(data.frame(class = 0, id = test.inds), path = "sample_solution.csv")


x = unlist(train[1, -ncol(train)]) / 255
i = as.cimg(array(x, c(32, 32, 1, 3)))
plot(i)
