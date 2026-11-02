#####################
#
#   Encode text on a character level and visualize 1d conv
#   Jann Goschenhofer, Compstat AG @ LMU
#   11/11/2018
#
###################

# read alphabet
alphabet = toString(read.table("./code/alphabet.txt")[1, 1])
alphList = strsplit(alphabet, split = "")[[1]]

# word to be encoded
word = " dummy review "
wordList = strsplit(word, split = "")[[1]]

enc = as.data.frame(matrix(data = 0, ncol = length(alphList), nrow = length(wordList)))
for (i in 1:length(wordList)) {
  enc[i, which(wordList[i] == alphList)] = 1
}

colnames(enc) = alphList

encPlot = function(x, ...){
  min <- min(x)
  max <- max(x)
  yLabels <- rownames(x)
  xLabels <- colnames(x)
  title <- c()
  # check for additional function arguments
  if ( length(list(...)) ){
    Lst <- list(...)
    if ( !is.null(Lst$zlim) ) {
      min <- Lst$zlim[1]
      max <- Lst$zlim[2]
    }
    if( !is.null(Lst$yLabels) ){
      yLabels <- c(Lst$yLabels)
    }
    if( !is.null(Lst$xLabels) ){
      xLabels <- c(Lst$xLabels)
    }
    if( !is.null(Lst$title) ){
      title <- Lst$title
    }
  }
  # check for null values
  if( is.null(xLabels) ){
    xLabels <- c(1:ncol(x))
  }
  if( is.null(yLabels) ){
    yLabels <- c(1:nrow(x))
  }

  layout(matrix(data=c(1,2), nrow=1, ncol=1), widths=c(4,1), heights=c(1,1))

  # Red and green range from 0 to 1 while Blue ranges from 1 to 0
  #ColorRamp <- rgb( seq(0,1,length=256),  # Red
  #  seq(0,1,length=256),  # Green
  #  seq(1,0,length=256))  # Blue
  ColorRamp = c("#ffffff", "#000000")
  # ColorLevels <- seq(min, max, length=length(ColorRamp))
  ColorLevels = c(0, 1)
  # Reverse Y axis
  reverse <- nrow(x) : 1
  yLabels <- yLabels[reverse]
  x <- x[reverse,]

  # Data Map
  par(mar = c(3,5,2.5,2), bg = NA)
  image(1:length(xLabels), 1:length(yLabels), t(x), col = ColorRamp, xlab="",
    ylab = "", axes = FALSE, zlim=c(min,max))
  if( !is.null(title) ) {
    title(main=title) 
  }
  axis(BELOW <- 1, at = 1:length(xLabels), labels=xLabels, cex.axis=2)
  axis(LEFT <- 2, at = 1:length(yLabels), labels=yLabels, las= HORIZONTAL<-1,
    cex.axis=2.5)

  grid(nx = ncol(x), ny = nrow(x))
  layout(1)
}


for (i in 1:(length(wordList) - 2)) {
  #i = 1
  Cairo::Cairo(file = paste0('./plots/text_encoding/', i, "_encoded_text.png"),
    type = "png")

  encPlot(as.matrix(enc[, 1:26]), zlim  = c(0, 1),
    yLabels = wordList, title = '')#paste0('Encoded text:', word))
  start = length(wordList) + -1.5 - i
  rect(xleft = 0.5, xright = 0.5 + 26, ybottom = start, ytop = start + 3, col = rgb(92/255, 178/255, 232/255, alpha = 0.2))
  dev.off()
}

#system("convert -delay 80 *filter.png cnnFilter.gif")
#file.remove(list.files(pattern = "filter.png"))





