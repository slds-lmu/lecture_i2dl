library(R.matlab)

d = readMat("caltech101_silhouettes_28.mat")

data = data.frame(y = c(d$Y), x = d$X)
set.seed(123)
data = data[sample(1:nrow(data)),]



test.inds = sample.int(8671, size = 1671)

train = data[-test.inds,]
test = data[test.inds,]
y = test$y
test$y = NULL
test$id = test.inds
write.csv(train, file = "train.csv")
write.csv(test, file = "test.csv")
write.csv(data.frame(y = y, id = test.inds), file = "solution.csv")

