# utils_test.jl

using Base.Test
using Utils

# Test case for preprocess_data function
@testset "preprocess_data test" begin
    # Define test data
    data = DataFrame(randn(100, 10))

    # Preprocess test data
    training_data, testing_data = Utils.preprocess_data(data)

    # Check if training_data and testing_data have correct dimensions
    @test size(training_data) == (70, 10)
    @test size(testing_data) == (30, 10)
end

# Test case for load_data function
@testset "load_data test" begin
    # Define test data
    file_path = "test_data.csv"
    data = DataFrame(randn(100, 10))
    CSV.write(file_path, data)

    # Load test data
    loaded_data = Utils.load_data(file_path)

    # Check if loaded_data is equal to test data
    @test all(data .== loaded_data)
end

# Test case for save_data function
@testset "save_data test" begin
    # Define test data
    file_path = "test_data.csv"
    data = DataFrame(randn(100, 10))

    # Save test data
    Utils.save_data(data, file_path)

    # Load saved data and check if it's equal to original test data
    loaded_data = Utils.load_data(file_path)
    @test all(data .== loaded_data)
end
