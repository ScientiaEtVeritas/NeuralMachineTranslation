from enum import IntEnum


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

    def addTokenList(self, tokens):
        for token in tokens:
            self.addToken(token)

    def addToken(self, token):
        if token not in self.token_index_map:
            self.token_index_map[token] = self.n_tokens
            self.token_count[token] = 1
            self.index_token_map[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token_count[token] += 1
            
    def filter_token(self, n):
        tokens_to_delete = [token for token, count in self.token_count.items() if count <= n]
        for token in tokens_to_delete:
            idx = self.token_index_map[token]
            del self.token_index_map[token]
            del self.token_count[token]  
            self.index_token_map[idx] = 'UNK'
                
        temp_map = {}
        for idx in range(len(self.index_token_map)):
            if self.index_token_map[idx] != 'UNK' or idx == LanguageTokens.UNK:
                temp_map[len(temp_map)] = self.index_token_map[idx]
        
        self.index_token_map = temp_map
        self.n_tokens = len(self.index_token_map)