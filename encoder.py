import torch
import torch.nn as nn
import rnn_utils


class EncoderRNN(nn.Module):
    def __init__(self, model_config, device=None):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = model_config.hidden_size
        self.embedding = nn.Embedding(
            model_config.input_size,
            model_config.hidden_size)
        self.bidirectional = model_config.bidirectional
        self.rnn_type = model_config.rnn_type
        self.num_layers = model_config.num_layers_encoder
        self.rnn = rnn_utils.initRNN(
            model_config.rnn_type,
            self.hidden_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=self.bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        return self.rnn(embedded, hidden)

    def initHidden(self):
        return torch.zeros(
            (2 if self.bidirectional else 1) *
            self.num_layers,
            1,
            self.hidden_size,
            device=self.device)

    def initEncoderHidden(self):
        if self.rnn_type == 'lstm':
            return (self.initHidden(), self.initHidden())
        elif self.rnn_type == 'gru':
            return self.initHidden()
