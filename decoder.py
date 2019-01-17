import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ModelConfig
import rnn_utils

class DecoderRNN(nn.Module):
    def __init__(self, model_config, device = None):
        super(DecoderRNN, self).__init__()
        self.device = device
        
        self.bidirectional = model_config.bidirectional
        self.hidden_size = model_config.hidden_size * (2 if self.bidirectional else 1)
        self.attention = model_config.attention
        self.rnn_type = model_config.rnn_type
        self.max_length = model_config.max_length

        self.embedding = nn.Embedding(model_config.output_size, self.hidden_size)
       
        if self.attention:
            if self.attention == 'local':
                self.attention_weights_linear = nn.Linear(self.hidden_size * 2, self.max_length) # attention_weights_linear(embedded[0], hidden[0])
                
            self.attention_combine_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
                
        self.dropout = nn.Dropout(model_config.dropout_p)

        self.rnn = rnn_utils.initRNN(self.rnn_type, self.hidden_size, self.hidden_size, model_config.num_layers_decoder) 

        self.out = nn.Linear(self.hidden_size, model_config.output_size)

    def forward(self, input, hidden, encoder_outputs = None):
        # input: Decoder Output (Init: SOS, ...)
        # Hidden: Tuple of Context Vector / Cell State of Decoder and Hidden State
        output = self.embedding(input).view(1, 1, -1) # output: Tuple of Hidden State and Cell State
        output = self.dropout(output)

        if not self.attention:
            attention_weights = None        
        
        if self.attention == 'local':
            if self.rnn_type == 'lstm':
                _hidden = hidden[0][0]
            elif self.rnn_type == 'gru':
                _hidden = hidden[0]
                
            attention_weights = F.softmax(self.attention_weights_linear(torch.cat((output[0], _hidden), 1)), dim = 1)
            attention_weighted_input = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            output = torch.cat((output[0], attention_weighted_input[0]), dim = 1)
            output = self.attention_combine_linear(output).unsqueeze(0)
        
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
                
        if self.attention == 'global':
            attention_weights = torch.empty(size=(encoder_outputs.size(0),))
            for i, encoder_output in enumerate(encoder_outputs):
                attention_weights[i] = torch.dot(output.squeeze(), encoder_output.squeeze())
                
            #Confine attention_weight to [0,1]
            attention_weights = F.softmax(attention_weights, dim=0)
            attention_context = torch.mm(attention_weights.reshape([1,-1]).to(self.device), encoder_outputs)
            attention_context = torch.cat((attention_context.squeeze(), output.squeeze()))
            output = torch.tanh(self.attention_combine_linear(attention_context))
            output = output.unsqueeze(0).unsqueeze(0)
                        
        output = F.log_softmax(self.out(output[0]), dim = 1)
        
        return output, hidden, attention_weights # attention_weights is None if attention not used

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
    def getDecoderHidden(self, encoder_hidden):
        if self.bidirectional and self.rnn_type == 'lstm':
            return (encoder_hidden[0].reshape((1,1,self.hidden_size)), encoder_hidden[1].reshape((1,1,self.hidden_size)))
        elif self.bidirectional and self.rnn_type == 'gru':
            return encoder_hidden.reshape((1,1,self.hidden_size))
        return encoder_hidden