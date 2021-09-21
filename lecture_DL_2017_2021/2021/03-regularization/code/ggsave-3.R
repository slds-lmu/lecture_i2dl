
########################################### dropout example
require("ggplot2")
require("reshape2")

mnist_dropout_dropoutTrain = read.csv("code/mnist_dropout_dropoutTrain", header = TRUE, check.names = FALSE)
dropoutTrain = melt(mnist_dropout_dropoutTrain, id = "epoch")

#subset dataset
#dropoutTrain = dropoutTrain[which(dropoutTrain$variable %in% c("(0;0)","(0;0.2)","(0;0.5)","(0.2;0)","(0.2;0.2)","(0.2;0.5)","(0.6;0)","(0.6;0.2)","(0.6;0.5)")),]
dropoutTrain = dropoutTrain[which(dropoutTrain$variable %in% c("(0;0)","(0.2;0.2)","(0.6;0.5)")),]

ggplot(data = dropoutTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "training error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
                     name = "epochs") + theme(legend.title.align = 0.5) +
  labs(colour = "Dropout rate:\n(Input; Hidden Layers)") +
  theme_bw()

############################################ dropout example

require("ggplot2")
require("reshape2")

mnist_dropout_dropoutTest = read.csv("code/mnist_dropout_dropoutTest", header = TRUE, check.names = FALSE)
# dropoutTest = mnist_dropout_dropoutTest[, -c(8:25)]

#try new
dropoutTest = mnist_dropout_dropoutTest[, c(1,2,10,25)]

dropoutTest = melt(dropoutTest, id = "epoch")

#subset
# dropoutTest1 = dropoutTest[1:200,c("epoch", "(0;0)","(0;0.2)","(0;0.5)")]
#dropoutTest = dropoutTest[1:200,c(epoch, (0;0),(0;0.2),(0;0.5))]

ggplot(data = dropoutTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.05)) + 
  scale_x_continuous(labels = function (x) floor(x), 
                     name = "epochs", limits = c(0, 200)) + theme(legend.title.align = 0.5) +
  labs(colour = "Dropout rate:\n(Input; Hidden Layers)") +
  theme_bw()

######################################## dropout or weight decay

require("ggplot2")

errorPlot = read.csv("code/mnist_visualization_model_total_error", header = TRUE)

ggplot(data = errorPlot, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.2)) + 
  scale_x_continuous(labels = function (x) floor(x), 
                     name = "epochs", limits = c(0, 500)) + theme(legend.title.align = 0.5) +
  labs(colour = "Test error \n comparison") +
  theme_bw()

############################################

