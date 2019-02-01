from enum import IntEnum
import itertools


class LanguageTokens(IntEnum):
    SOS = 0
    EOS = 1
    UNK = 2


class LanguageModel:
    def __init__(self):
        self.token_index_map = {}  # word → index
        self.token_count = {}
        self.index_token_map = {
            LanguageTokens.SOS: "SOS",
            LanguageTokens.EOS: "EOS",
            LanguageTokens.UNK: "UNK"}  # index → word
        self.n_tokens = 3  # Count SOS, EOS and UNK

    def addTokenList(self, list_tokens, filter_value):
        token_list = list(itertools.chain(*list_tokens))
        for token in token_list:
            self.count_token(token)
        for token in token_list:
            self.addToken(token, filter_value)
            
    def count_token(self, token):
        if token not in self.token_count:
            self.token_count[token] = 1
        else:
            self.token_count[token] += 1

    def addToken(self, token, filter_value):
        if token in self.token_count and self.token_count[token] > filter_value and token not in self.token_index_map:
            self.token_index_map[token] = self.n_tokens
            self.index_token_map[self.n_tokens] = token
            self.n_tokens += 1