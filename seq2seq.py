import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from LanguageModel import LanguageTokens

class seq2seq():
    def __init__(self, input_size, hidden_size, output_size, device = None, learning_rate = 0.01, teacher_forcing_ratio = 0.5):
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = EncoderRNN(input_size, hidden_size, device = device).to(device)
        self.decoder = DecoderRNN(hidden_size, output_size, encoder_bidirectional=self.encoder.bidirectional, device = device).to(device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()

    def train(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        encoder_hidden = self.encoder.initEncoderHidden()

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)
            
        decoder_input = torch.tensor([[LanguageTokens.SOS]], device=self.device)

        decoder_hidden = self.decoder.getDecoderHidden(encoder_hidden)
        
        loss = 0

        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        output_sentence = []

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            output_sentence.append(topi.item())

            if use_teacher_forcing: # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[di]  # Teacher forcing
            else: # Without teacher forcing: use its own predictions as the next input
                decoder_input = topi.squeeze().detach()  # detach from history as input
            #print("Training", di, criterion(decoder_output, target_tensor[di]))
            loss += self.criterion(decoder_output, target_tensor[di])

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

            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)

            decoder_input = torch.tensor([[LanguageTokens.SOS]], device=self.device)

            decoder_hidden = self.decoder.getDecoderHidden(encoder_hidden)

            output_sentence = []        
            loss = 0

            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                output_sentence.append(topi.item())

                loss += self.criterion(decoder_output, target_tensor[di])

                if topi.item() == LanguageTokens.EOS:
                    break

                decoder_input = topi.squeeze().detach()

            return loss.item(), torch.Tensor([output_sentence])
        
    def predict(self, input_tensor):
        with torch.no_grad():
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[i],
                                                         encoder_hidden)

            decoder_input = torch.tensor([[LanguageTokens.SOS]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            output_sentence = []        

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                output_sentence.append(topi.item())

                if topi.item() == LanguageTokens.EOS:
                    break

                decoder_input = topi.squeeze().detach()

            return decoded_words, torch.Tensor([output_sentence])

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
    def __init__(self, hidden_size, output_size, rnn_type = 'lstm', encoder_bidirectional = False, num_layers = 1, device = None):
        super(DecoderRNN, self).__init__()
        self.device = device
        hidden_size = hidden_size * (2 if encoder_bidirectional else 1)
        self.hidden_size = hidden_size
        self.encoder_bidirectional = encoder_bidirectional

        self.embedding = nn.Embedding(output_size, hidden_size)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
    def getDecoderHidden(self, encoder_hidden):
        if self.encoder_bidirectional and self.rnn_type == 'lstm':
            return (encoder_hidden[0].reshape((1,1,self.hidden_size)), encoder_hidden[1].reshape((1,1,self.hidden_size)))
        elif self.encoder_bidirectional and self.rnn_type == 'gru':
            return encoder_hidden.reshape((1,1,self.hidden_size))
        return encoder_hidden