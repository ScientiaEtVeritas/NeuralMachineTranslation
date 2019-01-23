from LanguageModel import LanguageTokens, LanguageModel
import torch


class DataLoader:
    def __init__(
            self,
            dataset,
            languages,
            max_length=50,
            languageModels=None,
            device=None):
        self.dataset = dataset
        self.languages = languages
        self.max_length = max_length
        self.loadFiles()
        if languageModels is not None:
            self.languageModels = languageModels
        else:
            self.languageModels = {
                self.languages[0]: LanguageModel(),
                self.languages[1]: LanguageModel()}
            self.prepareLanguageModels()
        self.device = device

    def loadFiles(self):
        # tuple of source sentences and target sentences
        self.data = self._preprocess((self._loadFile(self.languages[0]), self._loadFile(self.languages[1])))

    def __len__(self):
        return len(self.data[0])

    def _loadFile(self, l):
        return open(f'data/{self.dataset}.{l}', encoding='utf-8').read().lower().split('\n')

    def _preprocess(self, data):
        def valid_sentence(sentence):
            return len(sentence.split(' ')) <= self.max_length

        filtered = [(a.split(' '), b.split(' ')) for a, b in zip(*data) if valid_sentence(a) and valid_sentence(b)]

        filtered = zip(*filtered)
        return tuple([list(x) for x in filtered])

    def prepareLanguageModels(self):
        for i in range(len(self)):
            self.languageModels[self.languages[0]
                                ].addTokenList(self.data[0][i])
            self.languageModels[self.languages[1]
                                ].addTokenList(self.data[1][i])

    def _indexesFromSentence(self, lm, tokens):
        return [lm.token_index_map[token]
                if token in lm.token_index_map else LanguageTokens.UNK for token in tokens]

    def _tensorFromSentence(self, lm, tokens):
        indexes = self._indexesFromSentence(lm, tokens)
        indexes.append(LanguageTokens.EOS)
        return torch.tensor(indexes, dtype=torch.long,
                            device=self.device).view(-1, 1)

    def tensorsFromPos(self, pos):
        return tuple(self._tensorFromSentence(
            self.languageModels[self.languages[i]], self.data[i][pos]) for i in (0, 1))

    def real_estimated_sentence(
            self,
            real_target_tensor,
            estimated_target_tensor):
        real_target_sentence = " ".join(self.sentenceFromTensor(self.languages[1], real_target_tensor))
        estimated_target_sequence = " ".join(self.sentenceFromTensor(self.languages[1], estimated_target_tensor)[1:])
        return real_target_sentence, estimated_target_sequence

    def sentenceFromTensor(self, language, tensor):
        return [self.languageModels[language].index_token_map[int(word)] for word in tensor]
