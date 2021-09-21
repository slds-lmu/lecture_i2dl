# function to create high cost saddle points and low cost minima

require("ggplot2")
p <- function(x){
  return(
      ifelse(x >= 0 & x <= 4, 
        -x + 6,
        2))}

ggplot(data.frame(x=c(0, 12.5)), y=c(-5,6), aes(x=x)) + 
  geom_path(stat="function", fun=p) +
  scale_y_continuous(labels = function (x) floor(x), name = "learning rate", limits = c(0, 7), breaks = NULL) +
  scale_x_continuous(name = "training iteration", limits = c(0, 10), breaks = c(0:10)) +
  scale_colour_identity("Function", guide="legend") +
  annotate("text", x = 0, y = 6.3, label = "alpha[0]", parse = TRUE) +
  annotate("text", x = 1, y = 5.3, label = "alpha[1]", parse = TRUE) +
  annotate("text", x = 2, y = 4.3, label = "alpha[2]", parse = TRUE) +
  annotate("text", x = 3, y = 3.3, label = "alpha[3]", parse = TRUE) +
  annotate("text", x = 4, y = 2.3, label = "alpha[4]", parse = TRUE) + 
  annotate("text", x = 10, y = 2.3, label = "alpha[k+1]", parse = TRUE)












