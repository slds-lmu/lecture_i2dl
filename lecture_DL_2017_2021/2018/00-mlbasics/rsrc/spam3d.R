library(mlr)
library(plotly)


spam = getTaskData(spam.task)

# we choose some points for demonstration purposes
idx = which(spam$type == "spam" & spam$charExclamation <= 0.5 & spam$capitalAve <= 5 & spam$charExclamation > 0)[5:15]
idx = c(idx, which(spam$type == "nonspam" & spam$charExclamation <= 0.5 & spam$capitalAve <= 5 & spam$charExclamation > 0)[5:15])

df = spam[idx, ]
p = plot_ly(df, x = ~ charExclamation, y = ~ capitalAve, z = ~ your, color = ~ type, colors = c('#F8766D', '#619CFF')) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Frequency of Exclamation marks', dtick = 0.25, range = c(0, 0.5)),
                     yaxis = list(title = 'Longest sequence of capital letters', dtick = 2.5, range = c(0, 5)),
                     zaxis = list(title = 'Frequency of word free', dtick = 1, range = c(0, 2))))
p

# I do not now how to save plotly images as static .png
# need to find out