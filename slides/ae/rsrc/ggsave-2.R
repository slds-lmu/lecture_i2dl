

z = c(784, 256, 64, 32, 16, 8, 4, 2, 1)

input = mx.symbol.Variable("data") # mnist with 28x28 = 784
encoder = mx.symbol.FullyConnected(input, num_hidden = z[i])
decoder = mx.symbol.FullyConnected(encoder, num_hidden = 784)
activation = mx.symbol.Activation(decoder, "sigmoid")
output = mx.symbol.LinearRegressionOutput(activation)

model = mx.model.FeedForward.create(output,
                                    X = train.x, y = train.x,
                                    num.round = 50, 
                                    array.batch.size = 32,
                                    optimizer = "adam",
                                    initializer = mx.init.uniform(0.01), 
                                    eval.metric = mx.metric.mse
)