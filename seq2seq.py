import torch
import torch.nn as nn
from torch import optim
import random

from model_config import ModelConfig
from encoder import EncoderRNN
from decoder import DecoderRNN
from LanguageModel import LanguageTokens

class seq2seq():
    def __init__(self, model_config = None, state_dict = None, device = None):
        self.encoder = EncoderRNN(model_config, device = device).to(device)
        self.decoder = DecoderRNN(model_config, device = device).to(device)
        
        if state_dict:
            self.encoder.load_state_dict(state_dict['encoder'])
            self.decoder.load_state_dict(state_dict['decoder'])

        self.teacher_forcing_ratio = model_config.teacher_forcing_ratio
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=model_config.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=model_config.learning_rate)

        if state_dict:
            self.encoder_optimizer.load_state_dict(state_dict['encoder_optimizer'])
            self.decoder_optimizer.load_state_dict(state_dict['decoder_optimizer'])        
        
        self.criterion = nn.NLLLoss()
        self.beam_width = model_config.beam_width
        self.device = device

    def state_dict(self):
        return {'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'encoder_optimizer': self.encoder_optimizer.state_dict(),
                'decoder_optimizer': self.decoder_optimizer.state_dict()}

    def train(self, input_tensor, target_tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        encoder_outputs, decoder_hidden, last_context = self._forward_helper(input_tensor)            
        
        decoder_input = self._emptySentenceTensor()
        loss = 0
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        output_sentence = []

        for i in range(target_tensor.size(0)):
            decoder_output, decoder_hidden, _, last_context = self.decoder(decoder_input, decoder_hidden, encoder_outputs, last_context)
            _, topi = decoder_output.topk(1)
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
    
    #Last element of the prediction might not be <EOS>? Should we check it in DataLoader -> sentenceFromTensor?
    def predict(self, input_tensor):
        with torch.no_grad():
            encoder_outputs, decoder_hidden, last_context = self._forward_helper(input_tensor)

            sequences = [(0.0, [self._emptySentenceTensor()], [], decoder_hidden, [])]
            
            for _ in range(self.decoder.max_length):
                beam_expansion = []
                for apriori_log_prob, sentence, decoder_outputs, decoder_hidden, attention_weights_list in sequences:
                    decoder_input = sentence[-1]
                    if(decoder_input.item() != LanguageTokens.EOS):
                        decoder_output, decoder_hidden, attention_weights, last_context = self.decoder(decoder_input, decoder_hidden, encoder_outputs, last_context)

                        log_probabilities, indexes = decoder_output.data.topk(self.beam_width)
                        
                        for i in range(len(log_probabilities)):
                            log_prob = log_probabilities[i]
                            index = indexes.squeeze()[i] # (1,)
                            index = index.view(1,-1) # (1,1)
                            
                            beam_expansion.append((apriori_log_prob + log_prob, sentence + [index], decoder_outputs + [decoder_output], decoder_hidden, attention_weights_list + [attention_weights]))
                    else:
                        beam_expansion.append((apriori_log_prob, sentence, decoder_outputs, decoder_hidden, attention_weights))

                sequences = sorted(beam_expansion, reverse=True, key = lambda x: x[0])[:self.beam_width]
            
            _, sentence, decoder_outputs, _, attention_weights = sequences[0] # best sequence
            return sentence, decoder_outputs, attention_weights
    
    def evaluate(self, input_tensor, target_tensor):
        with torch.no_grad():
            target_length = target_tensor.size(0)
            
            sequence, decoder_outputs, _ = self.predict(input_tensor = input_tensor)
            
            loss = sum([self.criterion(decoder_outputs[i], target_tensor[i]) for i in range(min(len(decoder_outputs), target_length))])

            return loss.item(), torch.Tensor([sequence])

    def _forward_helper(self, input_tensor):        
        encoder_hidden = self.encoder.initEncoderHidden()
      
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)             
        decoder_hidden = self.decoder.getDecoderHidden(encoder_hidden)
        last_context = self.decoder.initContext()

        return encoder_outputs.view(-1, self.decoder.hidden_size), decoder_hidden, last_context

    def _emptySentenceTensor(self):
        return torch.tensor([[LanguageTokens.SOS]], device=self.device)