from LanguageModel import LanguageTokens, LanguageModel
import torch
import os

class DataLoader:
    def __init__(self, dataset, languages, max_length = 50, languageModels = None, device = None):
        self.dataset = dataset
        self.languages = languages
        self.max_length = max_length
        self.loadFiles()
        if languageModels is not None:
            self.languageModels = languageModels
        else:
            self.languageModels = {self.languages[0] : LanguageModel(), self.languages[1]: LanguageModel()}
            self.prepareLanguageModels()
        self.device = device
    
    def loadFiles(self):
        #list of (list of words being in the same line of the given file)
        self.data = (self._preprocess(self._loadFile(self.languages[0])),
                self._preprocess(self._loadFile(self.languages[1])))
        
    def __len__(self):
        return len(self.data[0])
        
    def _loadFile(self, l):
        return open(f'data/{self.dataset}.{l}', encoding='utf-8').read()
        
    def _preprocess(self, data):
        return [line.split(' ') for line in data.lower().split('\n') if len(line.split(' ')) <= self.max_length] # ggf. NUM_TOKEN, [] entfernen
    
    def prepareLanguageModels(self):
        for i in range(len(self)):
            self.languageModels[self.languages[0]].addTokenList(self.data[0][i])
            self.languageModels[self.languages[1]].addTokenList(self.data[1][i])
    
    def _indexesFromSentence(self, lm, tokens):
        return [lm.token_index_map[token] if token in lm.token_index_map else LanguageTokens.UNK for token in tokens ] # TODO: <UNK>
    
    def _tensorFromSentence(self, lm, tokens):
        #indexes = [SOS_token]
        indexes = self._indexesFromSentence(lm, tokens)
        indexes.append(LanguageTokens.EOS)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
    
    def tensorsFromPos(self, pos):
        return tuple(self._tensorFromSentence(self.languageModels[self.languages[i]], self.data[i][pos]) for i in (0,1))
        #input_tensor = self._tensorFromSentence(self.languageModels[self.languages[0]], self.data[0][pos])
        #target_tensor = self._tensorFromSentence(self.languageModels[self.languages[1]], self.data[1][pos])
        #return (input_tensor, target_tensor)    
        
    def sentenceFromTensor(self, real_target_tensor, estimated_target_tensor):
        print(f"estimated targ tnesor : {estimated_target_tensor}, {estimated_target_tensor.size()} ")
        real_target_sentence = " ".join([self.languageModels[self.languages[1]].index_token_map[int(word if isinstance(word, int) else word[0])] for word in real_target_tensor][:-1])
        estimated_target_sequence = [self.languageModels[self.languages[1]].index_token_map[int(word.item())] for word in estimated_target_tensor[0]]
        if estimated_target_sequence[-1] == LanguageTokens.EOS:
            estimated_target_sequence.pop()
        return real_target_sentence, " ".join(estimated_target_sequence[1:]) #First value is <SOS>