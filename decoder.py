import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ModelConfig
import rnn_utils

class DecoderRNN(nn.Module):
    def __init__(self, model_config, device = None, score = None):
        super(DecoderRNN, self).__init__()
        self.device = device
        
        self.bidirectional = model_config.bidirectional
        self.hidden_size = model_config.hidden_size * (2 if self.bidirectional else 1)
        self.attention = model_config.attention
        self.rnn_type = model_config.rnn_type
        self.max_length = model_config.max_length
        self.score = model_config.score
        self.num_layers = model_config.num_layers_decoder

        self.embedding = nn.Embedding(model_config.output_size, self.hidden_size)
       
        if self.attention:
            if self.score == 'MLP':
                self.attention_score_net = nn.Linear(self.hidden_size * 2, self.hidden_size) # attention_weights_linear(embedded[0], hidden[0])
                self.attention_score_parameters = nn.Parameter(torch.zeros(1, self.hidden_size, dtype=torch.float)) 
            self.attention_combine_linear = nn.Linear(self.hidden_size * 2, self.hidden_size) 
                
        self.dropout = nn.Dropout(model_config.dropout_p)
        #Because of the concatenation after embedding, if global_context, then input_size of rnn will be 2 * hidden_size
        self.rnn = rnn_utils.initRNN(self.rnn_type, self.hidden_size * (2 if self.attention =="global_context" else 1), self.hidden_size, self.num_layers) 

        self.out = nn.Linear(self.hidden_size, model_config.output_size)

    def forward(self, input, hidden, encoder_outputs = None, last_context = None):
        # input: Decoder Output (Init: SOS, ...)
        # Hidden: Tuple of Context Vector / Cell State of Decoder and Hidden State
        output = self.embedding(input).view(1, -1) # output: Tuple of Hidden State and Cell State
        output = self.dropout(output)

        if not self.attention:
            attention_weights = None        
                
        if self.attention == 'global_context':
            output = torch.cat((output, last_context), dim=1) #Take into account the last state of the context vector
        
        output, hidden = self.rnn(output.view(1, 1, -1), hidden)
        output = output.view(1,-1)
                
        if self.attention == 'global' or self.attention == 'global_context': 
            #Length of attention_weights = input_length
            input_length = encoder_outputs.size(0)
            attention_weights = torch.zeros(input_length)
            
            if self.score == "dot":
                attention_weights = torch.mm(output, encoder_outputs.t())
                
            elif self.score == "MLP":
                output_score = self.attention_score_net(torch.cat((output.repeat(input_length, 1), encoder_outputs), dim=1)) 
                output_score = torch.tanh(output_score) 
                attention_weights= torch.mm(self.attention_score_parameters, output_score.t())              

            attention_weights = F.softmax(attention_weights, dim=1)
            #Length of attention_weights = input_length
            new_context = torch.mm(attention_weights.to(self.device), encoder_outputs)
            attention_context = torch.cat((new_context, output), dim=1)
            output = torch.tanh(self.attention_combine_linear(attention_context))
                        
        output = F.log_softmax(self.out(output), dim = 1)
        if self.attention == 'global_context':
            return output, hidden, attention_weights, new_context
            
        return output, hidden, attention_weights, None # attention_weights is None if attention not used
    
    def getHidden(self, encoder_hidden):
        hidden_shape = (self.num_layers, 1, self.hidden_size)
        if self.bidirectional and self.rnn_type == 'lstm':
            return (encoder_hidden[0].reshape(hidden_shape), encoder_hidden[1].reshape(hidden_shape))
        elif self.bidirectional and self.rnn_type == 'gru':
            return encoder_hidden.reshape(hidden_shape)
        return encoder_hidden
    
    def initContext(self):
        return torch.zeros(1, self.hidden_size, device=self.device) if self.attention == "global_context" else None