import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from LanguageModel import LanguageTokens

class seq2seq():
    def __init__(self, input_size, hidden_size, output_size, device = None, learning_rate = 0.01, rnn_type = 'lstm', bidirectional = False, attention = 'global', max_length = 50, beam_width = 1, teacher_forcing_ratio = 0.5):
        self.device = device
        self.max_length = max_length
        self.beam_width = beam_width
        self.attention = attention
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = EncoderRNN(input_size, hidden_size, rnn_type = rnn_type, bidirectional = bidirectional, device = device).to(device)
        self.decoder = DecoderRNN(hidden_size, output_size, encoder_bidirectional=self.encoder.bidirectional, attention = attention, max_length = max_length, rnn_type = rnn_type, device = device).to(device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()

    def train(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        encoder_hidden = self.encoder.initEncoderHidden()
        
        if self.attention:
            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size * (2 if self.encoder.bidirectional else 1), device=self.device)

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)
            if self.attention:
                encoder_outputs[i] = encoder_output[0, 0]
            
        decoder_input = torch.tensor([[LanguageTokens.SOS]], device=self.device)

        decoder_hidden = self.decoder.getDecoderHidden(encoder_hidden)
        loss = 0

        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        output_sentence = []

        for i in range(target_length):
            if self.attention:
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
    
    def evaluate(self, input_tensor, target_tensor):
        with torch.no_grad():
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_hidden = self.encoder.initEncoderHidden()
            
            if self.attention:
                encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size * (2 if self.encoder.bidirectional else 1), device=self.device)

            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)
                if self.attention:
                    encoder_outputs[i] = encoder_output[0, 0]

            decoder_input = torch.tensor([[LanguageTokens.SOS]], device=self.device)

            decoder_hidden = self.decoder.getDecoderHidden(encoder_hidden)
            
            sequence, decoder_outputs = self.predict(input_tensor = input_tensor)
            
            loss = sum([self.criterion(decoder_outputs[i], target_tensor[i]) for i in range(min(len(decoder_outputs),target_length))])

            return loss.item(), torch.Tensor([sequence])
                    
    def predict(self, input_tensor):
        with torch.no_grad():
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initEncoderHidden()

            if self.attention:
                encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size * (2 if self.encoder.bidirectional else 1), device=self.device)

            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)
                
                if self.attention:
                    encoder_outputs[i] = encoder_output[0, 0]

            sequences = [(0.0, [torch.tensor([[LanguageTokens.SOS]], device=self.device)], [], self.decoder.getDecoderHidden(encoder_hidden))]
            
            for l in range(self.max_length):
                beam_expansion = []
                for apriori_log_prob, sentence, decoder_outputs, decoder_hidden in sequences:
                    decoder_input = sentence[-1]
                    if(decoder_input.item() != LanguageTokens.EOS):
                        if self.attention:
                            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden,encoder_outputs)
                        else:
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                        log_probabilities, indexes = decoder_output.data.topk(self.beam_width)
                        
                        for i in range(len(log_probabilities)):
                            log_prob = log_probabilities[i]
                            index = indexes.squeeze()[i] # (1,)
                            index = index.view(1,-1) # (1,1)
                            beam_expansion.append((apriori_log_prob + log_prob, sentence + [index], decoder_outputs + [decoder_output], decoder_hidden))
                    else:
                        beam_expansion.append((apriori_log_prob, sentence, decoder_outputs, decoder_hidden))

                sequences = sorted(beam_expansion, reverse=True, key = lambda x: x[0])[:self.beam_width]
                
            return sequences[0][1], sequences[0][2] # 0 best sequence, 1 sentence, 2 decoder_outputs

class EncoderRNN(nn.Module):
    # input_size: Größe des Vokabulars (One-Hot-Encoding)
    # hidden_size: Größe des Embedding-Vektors (Ein- und Ausgabegröße der RNN-Einheit)
    # https://isaacchanghau.github.io/post/lstm-gru-formula/
    def __init__(self, input_size, hidden_size, rnn_type = 'lstm', bidirectional = False, num_layers = 1, device = None):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_size, device=self.device)
    
    def initCellState(self):
        return torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_size, device=self.device)
    
    def initEncoderHidden(self):
        if self.rnn_type == 'lstm':
            return (self.initHidden(), self.initCellState())
        elif self.rnn_type == 'gru':
            return self.initHidden()
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, rnn_type = 'lstm', encoder_bidirectional = False, attention = False, dropout_p=0.1, max_length=40, num_layers = 1, device = None):
        super(DecoderRNN, self).__init__()
        self.device = device
        hidden_size = hidden_size * (2 if encoder_bidirectional else 1)
        self.hidden_size = hidden_size
        self.encoder_bidirectional = encoder_bidirectional
        self.attention = attention
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(output_size, hidden_size)
        
        if self.attention == 'local':
            self.attention_weights_linear = nn.Linear(self.hidden_size * 2, self.max_length) # attention_weights_linear(embedded[0], hidden[0])
            self.attention_combine_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif self.attention == 'global':
            self.attention_context_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        self.dropout = nn.Dropout(self.dropout_p)

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = num_layers)
        self.out = nn.Linear(hidden_size, output_size)

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
            output = torch.tanh(self.attention_context_linear(attention_context))
            output = output.unsqueeze(0).unsqueeze(0)
                        
        output = F.log_softmax(self.out(output[0]), dim = 1)
        
        if self.attention:
            return output, hidden, attention_weights
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
    def getDecoderHidden(self, encoder_hidden):
        if self.encoder_bidirectional and self.rnn_type == 'lstm':
            return (encoder_hidden[0].reshape((1,1,self.hidden_size)), encoder_hidden[1].reshape((1,1,self.hidden_size)))
        elif self.encoder_bidirectional and self.rnn_type == 'gru':
            return encoder_hidden.reshape((1,1,self.hidden_size))
        return encoder_hidden