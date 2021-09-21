# function to create cliff

require("ggplot2")
p <- function(x){
  return(
      ifelse(x >= 0 & x <= 5, 
        (-x+2.5)^3+200,
      ifelse(x >= 5 & x <= 10, 
        (-x+7.5)^3+184.375-(100-84.375),
      ifelse(x >= 10 & x <= 15, 
        (-x+12.5)^3+168.75-2*(84.375-68.75),
      ifelse(x >= 15 & x <= 20, 
       (-x+17.5)^3+153.125-3*(68.75-53.125),
       
      ifelse(x >= 20 & x <= 21, 
        -50*(x - 20) + 90.625,
      ifelse(x > 21 & x <= 22.5, 
        30*(x - 21.8)^2 + 20,
      ifelse(x > 22.5 & x <= 24, 
        20*-(x - 23)^2 + 40,
        
      ifelse(x > 24 & x <= 27, 
        30*(x - 24.8)^2 + 0,
      # ifelse(x > 27 & x <= 27.5, 
      #   10*-(x - 26)^2 + 20,
       
       
        0)))))))))}

ggplot(data.frame(x=c(0, 27)), aes(x=x)) + 
  geom_path(stat="function", fun=p) +
  scale_y_continuous(labels = function (x) floor(x), name = "cost", breaks = NULL) +
  scale_x_continuous(name = "", breaks = NULL) +
  scale_colour_identity("Function", guide="legend", 
                        labels = c("E(pi)"))












