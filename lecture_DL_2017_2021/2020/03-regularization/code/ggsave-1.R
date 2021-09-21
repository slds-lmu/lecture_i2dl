

require("ggplot2")

wdTrain = read.csv("code/mnist_weight_decay_wdTrain", header = TRUE)
options(scipen=999)
#wdTrain$variable = factor(wdTrain$variable)
wdTrain$variable = factor(wdTrain$variable, labels = c("0","10^(-5)","10^(-4)","10^(-3)","10^(-2)") )


ggplot(data = wdTrain, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "train error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
                     name = "epochs") + 
  labs(colour = "weight decay")

###############################################

wdTest = read.csv("code/mnist_weight_decay_wdTest", header = TRUE)
options(scipen=999)
wdTest$variable = factor(wdTest$variable, labels = c("0","10^(-5)","10^(-4)","10^(-3)","10^(-2)"))

ggplot(data = wdTest, aes(x = epoch, y = value, colour = variable)) +
  geom_line() +
  scale_y_continuous(name = "test error", limits = c(0, 0.1)) + 
  scale_x_continuous(labels = function (x) floor(x), 
                     name = "epochs") + 
  labs(colour = "weight decay")

###############################################



