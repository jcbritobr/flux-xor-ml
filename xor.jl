using Plots, Flux


X = [0f0 0 1 1; 0 1 0 1]
y = [0f0 1 1 0]

xor_model = Chain(
    Dense(2, 2, sigmoid),
    Dense(2, 1, sigmoid)
)

params(xor_model)

params(xor_model)[1]
params(xor_model)[3]

loss_fn(a, b) = Flux.mse(xor_model(a), b) 

opt = ADAM(0.001)

N = 5000
loss = zeros(N)

for i in 1:N
    Flux.train!(loss_fn, params(xor_model), [(X, y)], opt)
    loss[i] = loss_fn(X, y)
    if i % 100 == 0
        println("loss $loss[i]")
    end
end


loss_fn(X, y)
xor_model(X)

plot(1:N, loss, xlabel="Epochs", ylabel="Loss (mse)")