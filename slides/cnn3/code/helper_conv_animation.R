
### Convolute two signals
# conv from signal processing
# https://matheplanet.com/default3.html?call=viewtopic.php?topic=230056&ref=https%3A%2F%2Fwww.google.com%2F

conv1 = function(f, g, i){
    res = 0
    for(j in 1:i){
        res = res + f[j]*g[(i+1) - j]
    }
    return(res)
}
conv_full = function(f, g){
    res = vector(mode = "numeric")
    for(i in 1:length(f)){
        res[i] = conv1(f = f, g = g, i = i)
    }
    return(res)
}

f = c(1, 2, 2, 1, 0, 0)
g = c(1, -1, 0, 0, -1, 1)
y = conv_full(f = f, g = g)
plot(x = seq(length(f)), y = f, type = "l", col = "blue", 
    ylab = "f(x), g(x)", xlab = "x", 
    xaxt = "n", bty = "n",
    ylim = c(min(f, g, y), max(f, g, y)))
lines(x = seq(length(g)), y = g, col = "orange")
lines(x = seq(length(y)), y = y, col = "red")
axis(side = 1, pos = 0)
legend("bottom", legend = c("f(x)","g(x)", "y(x)"), xpd = TRUE, 
    col = c("blue", "orange", "red"), bty = "n", pch = 16, horiz = TRUE)

### Convolute with a filter kernel
# example from cs1114 class
# conv value on x
hi = function(f, g, i){
    m = length(g)
    res = 0
    for(j in 1:m){
        idx = ceiling(i - j + m/2)
        if(idx > 0 & idx <= length(f)){
            res = res + g[j]*f[idx]
            
        }
    }
    return(res)
}
# conv value whole f
h = function(f, g){
    res = vector(mode = "numeric", length = length(f))
    for(l in 1:length(res)){
        res[l] = hi(f = f, g = g, i = l)
    }
    return(res)
}

## plot
# enter values for input and filter
kwidth = 5
f = c(0, 0, 0, 0, 0, 0, 1, 5, 6, 4, 2, 3, 4, 4, 4, 5, 6, 3, 3, 2, 4, 6, 1, 0, 0, 0, 0)
#f = c(0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0)
g = rep(1/kwidth, kwidth)

for(i in kwidth:length(f)){
    png(filename = paste0("./plots/conv_animations/conv_anim_", i, ".png"), 
        bg = "transparent", width = 640)
    
    # f
    par(mar = c(5.1, 4.1, 2.1, 4.4))
    plot(x = seq(length(f)), y = f, type = "l", col = "blue", 
        ylab = "f(x), h(x)", xlab = "x", 
        #xaxt = "n", bty = "n", 
        ylim = c(0, max(f)))
    
    # h
    h_vec = vector(i, mode = "numeric")
    for(l in 1:i){
        h_vec[l] = hi(f = f, g = g, i = l)
    }
    lines(x = seq(i), y = h_vec, type = "l", col ="red")
    
    # g
    par(new=TRUE)
    plot(x = seq(length(f)), xaxt = "n", xlab = "", ylab ="", yaxt ="n", 
        ylim = c(0, 1/kwidth), axes = FALSE)
    
    rect(xleft = max(min((i-length(g) + 1):i), 1), xright = max((i-length(g) + 1):i), 
        ybottom = 0, ytop = max(g), col = rgb(255/255,165/255,0, 0.4), border = NA)
    
    axis(side = 4, at = seq(0, 0.2, 0.05))
    axis(side = 1, at = seq(1:length(f)))
    mtext("g(x)",side=4,line=3)
    # legend
    legend("bottom", legend = c("f(x)","g(x)", "h(x)"), xpd = TRUE,
        col = c("blue", "orange", "red"), bty = "n", pch = 16, horiz = TRUE)
    dev.off()
}




h5 = h_i(f = fx, g = gx, i = 5)
h5
plot(unlist(fx), type = 'l')


plot(unlist(fx), type = 'l')
for(i in kernel_size+1:output_size){
    # i = 2
    hx[i] = 0
    lines(unlist(hx))
    for(j in 1:kernel_size){
        # j = 2
        hx[i] = hx[i][[1]] + fx[i - j][[1]] * gx[j][[1]]
    }
}

plot(unlist(fx), type = 'l')
lines(unlist(y), type = 'l')





