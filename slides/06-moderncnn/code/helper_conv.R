###########
# Convolution and Cross Correlation animations
##########

# library used for simpson-rule-based integration
library(Bolstad2)

ft = function(t){
    as.numeric(t>-1) * as.numeric(t<2) * 1 
}

gt = function(t){
    # as.numeric(t>-2) * as.numeric(t<1) * 1 
    # max(0, 1 - abs(t))
    value = as.numeric(t >= 0) * (1 - 0.2*abs(t))
    return(ifelse(value > 0, value, 0))
}

gt2 = function(t){
    # hacky implement for xcorrel animation
    value = as.numeric(t < 1/0.2) * as.numeric(t > 0) * 0.2*abs(t)
    return(ifelse(value > 0, value, 0))
}


flipped = function(t0, tau){
    gt(t0-tau)
}

conv = function(t0, tau){
    ft(tau) * gt(t0-tau)
}

conv2 = function(t0, tau){
    ft(tau) * gt2(t0-tau)
}

xcorrel = function(t0, tau){
    ft(tau) * gt(t0+tau)
}



get_integral = function(t){
    Ht = conv(t0 = t, tau = input)
    integ = sintegral(input[1:which(input == t)], Ht[1:which(input == t)])$int
}

get_full_integral = function(t0){
    unlist(c(list(0), lapply(input[2:which(input == t0)], get_integral)))
}

get_integral2 = function(t){
    Ht = conv2(t0 = t, tau = input)
    integ = sintegral(input[1:which(input == t)], Ht[1:which(input == t)])$int
}

get_full_integral2 = function(t0){
    unlist(c(list(0), lapply(input[2:which(input == t0)], get_integral2)))
}

get_integral_xcorrel = function(t){
    Ht = xcorrel(t0 = t, tau = input)
    integ = sintegral(input[1:which(input == t)], Ht[1:which(input == t)])$int
}

get_full_integral_xcorrel = function(t0){
    unlist(c(list(0), lapply(input[2:which(input == t0)], get_integral_xcorrel)))
}


plot_xcorrel_step = function(f, g, t, input){
    # plots one xcorrelation step of f with g
    # over input range
    # also colors the convolved area aka the integral
    # Currently replaced by the hacke plot_conv_step2 variant
    
    gt_shift = function(t, input){
        gt(t+input)
    }
    
    Ht = xcorrel(t0 = t, tau = input)
    Gt = gt_shift(t = t, input = input)
    Ft = ft(input)
    
    par(mfrow = c(2, 1))
    # 1st
    plot(x = input, y = Ft, col = 'blue', type = 'l', 
        ylab = "f(x), g(x)", xlab  = 'x', main = "Cross correlation of f(x) with g(x)")
    lines(x = input, y = Gt, col = 'orange', type = 'l')
    lines(x = input, y = Ht, col = 'purple', type = 'l')
    polygon(x = input, y = Ht, col = 'purple')
    
    legend("topright", legend = c("f(x)","g(x)", "h(x)"), xpd = TRUE,
        col = c("blue", "orange", "purple"), bty = "n", pch = 16, horiz = FALSE)
    
    # 2nd
    plot(x = input, y = get_full_integral_xcorrel(t0 = max(input)), 
        col = 'purple', type = 'l', 
        ylab = 'h(x)', xlab  = 'x')
    points(x = t, y = get_integral(t), type = 'o', pch = 19, cex = 2, col = 'red')
}

plot_conv_step = function(f, g, t, input){
    # plots one convolution step of f with g
    # over input range
    # also colors the convolved area aka the integral
    
    gt_shift = function(t, input){
        gt(t-input)
    }
    
    Ht = conv(t0 = t, tau = input)
    
    Gt = gt_shift(t = t, input = input)
    Ft = ft(input)
    
    par(mfrow = c(2, 1))
    # 1st
    plot(x = input, y = Ft, col = 'blue', type = 'l', 
        ylab = "f(x), g(x)", xlab  = 'x', main = "Convolution of f(x) with g(x)")
    lines(x = input, y = Gt, col = 'orange', type = 'l')
    lines(x = input, y = Ht, col = 'purple', type = 'l')
    polygon(x = input, y = Ht, col = 'purple')
    
    legend("topright", legend = c("f(x)","g(x)", "h(x)"), xpd = TRUE,
        col = c("blue", "orange", "purple"), bty = "n", pch = 16, horiz = FALSE)
    
    # 2nd
    plot(x = input, y = get_full_integral(t0 = max(input)), 
        col = 'purple', type = 'l', 
        ylab = 'h(x)', xlab  = 'x')
    points(x = t, y = get_integral(t), type = 'o', pch = 19, cex = 2, col = 'red')
}

plot_conv_step2 = function(f, g, t, input){
    # plots one convolution step of f with g
    # over input range
    # also colors the convolved area aka the integral
    
    gt_shift = function(t, input){
        gt2(t-input)
    }
    
    Ht = conv2(t0 = t, tau = input)
    
    Gt = gt_shift(t = t, input = input)
    Ft = ft(input)
    
    par(mfrow = c(2, 1))
    # 1st
    plot(x = input, y = Ft, col = 'blue', type = 'l', 
        ylab = "f(x), g(x)", xlab  = 'x', main = "Cross-correlation of f(x) with g(x)")
    lines(x = input, y = Gt, col = 'orange', type = 'l')
    lines(x = input, y = Ht, col = 'purple', type = 'l')
    polygon(x = input, y = Ht, col = 'purple')
    
    legend("topright", legend = c("f(x)","g(x)", "h(x)"), xpd = TRUE,
        col = c("blue", "orange", "purple"), bty = "n", pch = 16, horiz = FALSE)
    
    # 2nd
    plot(x = input, y = get_full_integral2(t0 = max(input)), 
        col = 'purple', type = 'l', 
        ylab = 'h(x)', xlab  = 'x')
    points(x = t, y = get_integral2(t), type = 'o', pch = 19, cex = 2, col = 'red')
}


##############
# Convolution
#############

input = round(seq(-6.2, 8, 0.1), 1)

idx = 1
for(t in seq(-6, 8, 0.5)){
    png(filename = paste0("./plots/conv_animations/conv_anim_", idx, ".png"), 
        bg = "transparent", width = 640)
    idx = idx + 1
    plot_conv_step(f = ft, g = gt, t = t, input = input)
    dev.off()
}

##############
# Cross Correlation
#############

# hacky implement via flipped g function
# TODO: fix implementation via the xcorrel function
# But good enough for our purpose
# hope, the code is commented well enough, dear future-hiwi ;-)

idx = 1
for(t in seq(-6, 8, 0.5)){
    png(filename = paste0("./plots/xcorrel_animations/xcorrel_anim_", idx, ".png"), 
        bg = "transparent", width = 640)
    idx = idx + 1
    plot_conv_step2(f = ft, g = gt, t = t, input = input)
    dev.off()
}


#################
#   Static image
#################

input = round(seq(-6.2, 6, 0.1), 1)

Gt = gt(input)
Ft = ft(input)

png(filename = paste0("./plots/conv_animations/conv_static.png"), 
    bg = "transparent", width = 640, height = 320)
par(mfrow = c(1, 2))
plot(x = input, y = Ft, col = 'blue', 
    type = 'l', main = 'f(x)', 
    ylab = "f(x)", xlab = "x")
plot(x = input, y = Gt, col = 'orange', 
    type = 'l', main = 'g(x)', 
    ylab = "g(x)", xlab = "x")
dev.off()


# thx to https://dspillustrations.com/pages/posts/misc/convolution-examples-and-the-convolution-integral.html

# 
# 
# t0 = 0
# Gt = gt(input)
# Ft = ft(input)
# Ht = prod(t0 = t0, tau = input)
# integ = sintegral(input[1:which(input == t0)], Ht[1:which(input == t0)])$int
# integ
# plot(x = input, y = Gt, col = 'orange', 
#     type = 'l', main = 'f(x) to be convolved with g(x)', 
#     ylab = "Q(x)", xlab = "x")
# lines(x = input, y = Ft, col = 'blue', type = 'l')
# lines(x = input, y = Ht, col = 'purple', type = 'l')
# legend("topright", legend = c("f(x)","g(x)"), xpd = TRUE)
# 
# 
# 
# int_list = unlist(c(list(0), lapply(input[-1], get_integral)))
# foo = get_full_integral_xcorrel(t0 = 6)
# plot(foo)
# length(int_list)
# 
# t0 = 3
# 
# plot(x = input, y = foo, col = 'purple', type = 'l')
# int = get_integral(t0)
# points(x = t0, y = int, type = 'o', pch = 19, cex = 2, col = 'red')
# int
# 
# 
# a = get_integral(5.9)
# a
# 
# plot_conv_step(f = ft, g = gt, t = 1, input = input)
# 
# 
# n <- 100
# mean <- 50
# sd <- 50
# 
# x <- c(0, 0, 0, 0, 0, 0, 0, seq(20, 80, length=n), 0, 0, 0, 0)
# y <- dnorm(x, mean, sd) *100
# 
# plot(y, type = 'l')
# # using sintegral in Bolstad2
# require(Bolstad2)
# sintegral(x,y)$int
# 
# x = seq(1, 10, 1)
# y = c(0, 0, 0, 0, 1, 2, 3, 4, 4, 2)
# plot(x,y, type = 'l')
# 
# max = 8
# sintegral(x[1:max], y[1:max])
# 
# 
