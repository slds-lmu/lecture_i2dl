


fc1 = mx.symbol.FullyConnected(data, num_hidden = 512)
act1 = mx.symbol.Activation(fc1, activation = "relu")
fc2 = mx.symbol.FullyConnected(act1, num_hidden = 512)
act2 = mx.symbol.Activation(fc2, activation = "relu")
fc3 = mx.symbol.FullyConnected(act2, num_hidden = 512)
act3 = mx.symbol.Activation(fc3, activation = "relu")
fc4 = mx.symbol.FullyConnected(act3, num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")





fc1 = mx.symbol.FullyConnected(data, num_hidden = 512)
act1 = mx.symbol.Activation(fc1, act_type = "relu")
bn1 = mx.symbol.BatchNorm(act1)
fc2 = mx.symbol.FullyConnected(bn1, num_hidden = 512)
act2 = mx.symbol.Activation(fc2, act_type = "relu")
bn2 = mx.symbol.BatchNorm(act2)
fc3 = mx.symbol.FullyConnected(bn2, num_hidden = 512)
act3 = mx.symbol.Activation(fc3, act_type = "relu")
bn3 = mx.symbol.BatchNorm(act3)
fc4 = mx.symbol.FullyConnected(bn3, num_hidden = 10)
softmax = mx.symbol.SoftmaxOutput(fc4, name = "sm")