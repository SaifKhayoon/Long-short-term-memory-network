module Utils
    using CSV
    using DataFrames

    function preprocess_data(data)
        # Remove missing values
        data = DataFrame(dropmissing(data))

        # Normalize data
        data = DataFrame(normalize(data, :log))

        # Convert categorical variables to numerical variables
        categorical_columns = [:column1, :column2, :column3]
        for column in categorical_columns
            data[column] = categorical(data[column])
            data[column] = labelencode(data[column])
        end

        # Split data into training and testing sets
        training_data, testing_data = splitdata(data, at = 0.7)

        return training_data, testing_data
    end
    function load_data(file_path)
        data = CSV.read(file_path)
        return data
    end

    function save_data(data, file_path)
        CSV.write(file_path, data)
    end
end
