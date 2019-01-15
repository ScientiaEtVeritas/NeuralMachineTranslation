import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from LanguageModel import LanguageTokens

class ModelConfig():
    def __init__(self, input_size, hidden_size, output_size, max_length = 50, rnn_type = 'lstm', bidirectional = False, attention = 'global', dropout_p = 0.1, num_layers_encoder=1, num_layers_decoder=1):
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

class TrainingConfig():
    def __init__(self, learning_rate = 0.01, teacher_forcing_ratio = 0.5):
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio

class PredictionConfig():
    def __init__(self, beam_width = 1):
        self.beam_width = beam_width

class seq2seq():
    def __init__(self, model_config = None, training_config = None, prediction_config = None, state_dict = None, device = None):
        self.encoder = EncoderRNN(model_config, device = device).to(device)
        self.decoder = DecoderRNN(model_config, device = device).to(device)
        
        if not model_config:
            assert(state_dict)
            self.encoder.load_state_dict(state_dict['encoder'])
            self.decoder.load_state_dict(state_dict['decoder'])

        if training_config:         
            self.teacher_forcing_ratio = training_config.teacher_forcing_ratio
            self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=training_config.learning_rate)
            self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=training_config.learning_rate)
        else:
            assert(state_dict)
            self.teacher_forcing_ratio = state_dict['teacher_forcing_ratio']
            self.encoder_optimizer = optim.SGD()
            self.decoder_optimizer = optim.SGD()

            self.encoder_optimizer.load_state_dict(state_dict['encoder_optimizer'])
            self.decoder_optimizer.load_state_dict(state_dict['decoder_optimizer'])        
        
        self.criterion = nn.NLLLoss()

        self.prediction_config = prediction_config
        
        self.device = device

    def state_dict():
        return {'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'encoder_optimizer': self.encoder_optimizer.state_dict(),
                'decoder_optimizer': self.decoder_optimizer.state_dict(),
                'teacher_forcing_ratio': self.teacher_forcing_ratio}

    def train(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        target_length = target_tensor.size(0)
        
        encoder_outputs, decoder_hidden = self._forward_helper(input_tensor)
            
        decoder_input = torch.tensor([[LanguageTokens.SOS]], device=self.device)

        loss = 0

        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        output_sentence = []

        for i in range(target_length):
            if self.decoder.attention:
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            output_sentence.append(topi.item())

            if use_teacher_forcing: # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[i]  # Teacher forcing
            else: # Without teacher forcing: use its own predictions as the next input
                decoder_input = topi.squeeze().detach()  # detach from history as input
                
            loss += self.criterion(decoder_output, target_tensor[i])

            if decoder_input.item() == LanguageTokens.EOS:
                break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item(), torch.Tensor([output_sentence])
                        
    def predict(self, input_tensor):
        with torch.no_grad():
            input_length = input_tensor.size()[0]

            encoder_outputs, decoder_hidden = self._forward_helper(input_tensor)
            
            sequences = [(0.0, [torch.tensor([[LanguageTokens.SOS]], device=self.device)], [], decoder_hidden)]
            
            for l in range(self.decoder.max_length):
                beam_expansion = []
                for apriori_log_prob, sentence, decoder_outputs, decoder_hidden in sequences:
                    decoder_input = sentence[-1]
                    if(decoder_input.item() != LanguageTokens.EOS):
                        if self.decoder.attention:
                            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden,encoder_outputs)
                        else:
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                        log_probabilities, indexes = decoder_output.data.topk(self.prediction_config.beam_width)
                        
                        for i in range(len(log_probabilities)):
                            log_prob = log_probabilities[i]
                            index = indexes.squeeze()[i] # (1,)
                            index = index.view(1,-1) # (1,1)
                            beam_expansion.append((apriori_log_prob + log_prob, sentence + [index], decoder_outputs + [decoder_output], decoder_hidden))
                    else:
                        beam_expansion.append((apriori_log_prob, sentence, decoder_outputs, decoder_hidden))

                sequences = sorted(beam_expansion, reverse=True, key = lambda x: x[0])[:self.prediction_config.beam_width]
                
            return sequences[0][1], sequences[0][2] # 0 best sequence, 1 sentence, 2 decoder_outputs
    
    def evaluate(self, input_tensor, target_tensor):
        with torch.no_grad():
            target_length = target_tensor.size(0)
            
            sequence, decoder_outputs = self.predict(input_tensor = input_tensor)
            
            loss = sum([self.criterion(decoder_outputs[i], target_tensor[i]) for i in range(min(len(decoder_outputs),target_length))])

            return loss.item(), torch.Tensor([sequence])

    def _forward_helper(self, input_tensor):
        input_length = input_tensor.size(0)
        
        encoder_hidden = self.encoder.initEncoderHidden()
        
        if self.decoder.attention:
            encoder_outputs = torch.zeros(self.decoder.max_length, self.encoder.hidden_size * (2 if self.encoder.bidirectional else 1), device=self.device)

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)
            if self.decoder.attention:
                encoder_outputs[i] = encoder_output[0, 0]
                
        decoder_hidden = self.decoder.getDecoderHidden(encoder_hidden)
        
        return encoder_outputs, decoder_hidden

class EncoderRNN(nn.Module):
    # input_size: Größe des Vokabulars (One-Hot-Encoding)
    # hidden_size: Größe des Embedding-Vektors (Ein- und Ausgabegröße der RNN-Einheit)
    # https://isaacchanghau.github.io/post/lstm-gru-formula/
    def __init__(self, model_config = None, device = None):
        super(EncoderRNN, self).__init__()
        self.device = device
        if model_config:
            self.hidden_size = model_config.hidden_size
            self.embedding = nn.Embedding(model_config.input_size, model_config.hidden_size)
            self.bidirectional = model_config.bidirectional
            self.rnn_type = model_config.rnn_type
            
            if model_config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional = self.bidirectional, num_layers = model_config.num_layers_encoder)
            elif model_config.rnn_type == 'gru':
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size, bidirectional = self.bidirectional, num_layers = model_config.num_layers_encoder)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_size, device=self.device)
        
    def initEncoderHidden(self):
        if self.rnn_type == 'lstm':
            return (self.initHidden(), self.initHidden())
        elif self.rnn_type == 'gru':
            return self.initHidden()
        
class DecoderRNN(nn.Module):
    def __init__(self, model_config = None, device = None):
        super(DecoderRNN, self).__init__()
        self.device = device
        if model_config:
            self.bidirectional = model_config.bidirectional
            self.hidden_size = model_config.hidden_size * (2 if self.bidirectional else 1)
            self.attention = model_config.attention
            self.rnn_type = model_config.rnn_type
            self.max_length = model_config.max_length

            self.embedding = nn.Embedding(model_config.output_size, self.hidden_size)
            
            if self.attention == 'local':
                self.attention_weights_linear = nn.Linear(self.hidden_size * 2, self.max_length) # attention_weights_linear(embedded[0], hidden[0])

            if self.attention:
                self.attention_combine_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
                
            self.dropout = nn.Dropout(model_config.dropout_p)

            if self.rnn_type == 'lstm':
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers = model_config.num_layers_decoder)
            elif self.rnn_type == 'gru':
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers = model_config.num_layers_decoder)
            self.out = nn.Linear(self.hidden_size, model_config.output_size)

    def forward(self, input, hidden, encoder_outputs = None):
        # input: Decoder Output (Init: SOS, ...)
        # Hidden: Tuple of Context Vector / Cell State of Decoder and Hidden State
        output = self.embedding(input).view(1, 1, -1) # output: Tuple of Hidden State and Cell State
        output = self.dropout(output)
    
        if self.attention == 'local':
            if self.rnn_type == 'lstm':
                attention_weights = F.softmax(self.attention_weights_linear(torch.cat((output[0], hidden[0][0]), 1)), dim = 1)
            elif self.rnn_type == 'gru':
                attention_weights = F.softmax(self.attention_weights_linear(torch.cat((output[0], hidden[0]), 1)), dim = 1)
            attention_weighted_input = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            output = torch.cat((output[0], attention_weighted_input[0]), dim = 1)
            output = self.attention_combine_linear(output).unsqueeze(0)
        
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
                
        if self.attention == 'global':
            attention_weights = torch.empty(size=(encoder_outputs.size(0),))
            for i, encoder_output in enumerate(encoder_outputs):
                attention_weights[i] = torch.dot(output.squeeze(), encoder_output.squeeze())
            
            attention_context = torch.mm(attention_weights.reshape([1,-1]), encoder_outputs)
            attention_context = torch.cat((attention_context.squeeze(), output.squeeze()))
            output = torch.tanh(self.attention_combine_linear(attention_context))
            output = output.unsqueeze(0).unsqueeze(0)
                        
        output = F.log_softmax(self.out(output[0]), dim = 1)
        
        if self.attention:
            return output, hidden, attention_weights
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
    def getDecoderHidden(self, encoder_hidden):
        if self.bidirectional and self.rnn_type == 'lstm':
            return (encoder_hidden[0].reshape((1,1,self.hidden_size)), encoder_hidden[1].reshape((1,1,self.hidden_size)))
        elif self.bidirectional and self.rnn_type == 'gru':
            return encoder_hidden.reshape((1,1,self.hidden_size))
        return encoder_hidden