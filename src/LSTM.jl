module LSTM
    using Flux
    
    struct LSTMCell
        Wf::Array{Float32}
        bf::Array{Float32}
        Wi::Array{Float32}
        bi::Array{Float32}
        Wc::Array{Float32}
        bc::Array{Float32}
        Wo::Array{Float32}
        bo::Array{Float32}
        state::Array{Float32}
    end

    function LSTMCell(input_size, hidden_size)
        Wf = randn(hidden_size, input_size + hidden_size)
        bf = zeros(hidden_size)
        Wi = randn(hidden_size, input_size + hidden_size)
        bi = zeros(hidden_size)
        Wc = randn(hidden_size, input_size + hidden_size)
        bc = zeros(hidden_size)
        Wo = randn(hidden_size, input_size + hidden_size)
        bo = zeros(hidden_size)
        state = zeros(hidden_size)
        return LSTMCell(Wf, bf, Wi, bi, Wc, bc, Wo, bo, state)
    end

    function forward(lstm::LSTMCell, x, state)
        # calculate forget gate
        ft = sigmoid.(lstm.Wf * hcat(x, state) .+ lstm.bf)
        # calculate input gate
        it = sigmoid.(lstm.Wi * hcat(x, state) .+ lstm.bi)
        # calculate cell state candidate
        ct = tanh.(lstm.Wc * hcat(x, state) .+ lstm.bc)
        # calculate next cell state
        c_next = ft .* state + it .* ct
        # calculate output gate
        ot = sigmoid.(lstm.Wo * hcat(x, c_next) .+ lstm.bo)
        # calculate next hidden state
        h_next = ot .* tanh.(c_next)
        return h_next, c_next
    end
end
