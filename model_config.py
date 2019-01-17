class ModelConfig():
    def __init__(self, input_size, hidden_size, output_size, max_length = 50, rnn_type = 'lstm', bidirectional = False, attention = 'global', dropout_p = 0.1, num_layers_encoder=1, num_layers_decoder=1, learning_rate = 0.01, teacher_forcing_ratio = 0.5, beam_width = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.attention = attention
        self.dropout_p = dropout_p
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_width = beam_width