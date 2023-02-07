using Base.Test
using LSTM

# Test case for LSTM forward pass
@testset "LSTM forward pass test" begin
    # Define test inputs
    input = randn(100, 10)
    hidden = randn(100, 20)
    weights = randn(30, 20)

    # Compute LSTM forward pass
    output, new_hidden = LSTM.forward(input, hidden, weights)

    # Check if output has correct dimensions
    @test size(output) == (100, 30)

    # Check if new hidden state has correct dimensions
    @test size(new_hidden) == (100, 20)
end

# Test case for LSTM backward pass
@testset "LSTM backward pass test" begin
    # Define test inputs
    input = randn(100, 10)
    hidden = randn(100, 20)
    weights = randn(30, 20)
    grad_output = randn(100, 30)

    # Compute LSTM backward pass
    grad_input, grad_weights, grad_hidden = LSTM.backward(input, hidden, weights, grad_output)

    # Check if grad_input has correct dimensions
    @test size(grad_input) == (100, 10)

    # Check if grad_weights has correct dimensions
    @test size(grad_weights) == (30, 20)

    # Check if grad_hidden has correct dimensions
    @test size(grad_hidden) == (100, 20)
end
