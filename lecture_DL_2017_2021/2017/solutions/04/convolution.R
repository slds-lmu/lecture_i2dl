conv = function(m, kernel, stride = 1, padding = c("none", "same")) {

  m.x = nrow(m)
  m.y = ncol(m)
  range.x.left = floor((nrow(kernel) - 1)/2)
  range.x.right = ceiling((nrow(kernel) - 1)/2)
  range.y.up = floor((ncol(kernel) - 1)/2)
  range.y.down = ceiling((ncol(kernel) - 1)/2)

  if (padding == "none") {
    seq.x = seq(from = range.x.left + 1, to = m.x - range.x.right, by = stride)
    seq.y = seq(from = range.y.up + 1, to = m.y - range.y.down, by = stride)
  } else {
    seq.x = seq(from = range.x.left + 1, to = m.x, by = stride)
    seq.y = seq(from = range.y.up + 1, to = m.x, by = stride)
    m.pad = matrix(0, nrow = m.x + range.x.left + range.x.right, ncol = m.y + range.y.down + range.y.up)
    m.pad[(range.x.left + 1):(nrow(m.pad) - (range.x.right)), (range.y.up + 1):(ncol(m.pad) - (range.y.down))] = m
    m = m.pad
  }

  feature.map = matrix(NA, ncol = length(seq.y), nrow = length(seq.x))

  for(i in seq_along(seq.x)) {
    for(j in seq_along(seq.y)) {
      patch = m[(seq.x[i] - range.x.left):(seq.x[i] + range.x.right), (seq.y[j] - range.y.up):(seq.y[j] + range.y.down)]
      feature.map[i,j] = sum(patch * kernel)
    }
  }
  return(feature.map)

}


m = matrix(1:9, ncol = 3)
kernel = matrix(c(-1, 0, 0, 0), ncol = 2)

conv(m, kernel, padding = "none", stride = 1)
conv(m, kernel, padding = "none", stride = 2)

conv(m, kernel, padding = "same", stride = 1)
conv(m, kernel, padding = "same", stride = 2)


m = matrix(1:16, ncol = 4)
kernel = matrix(c(-1, 0, 0, 0, 0, 0, 0, 0, 0), ncol = 3)

conv(m, kernel, padding = "none", stride = 1)
conv(m, kernel, padding = "none", stride = 2)

conv(m, kernel, padding = "same", stride = 1)
conv(m, kernel, padding = "same", stride = 2)
