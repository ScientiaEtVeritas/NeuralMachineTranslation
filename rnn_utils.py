import torch.nn as nn

def initRNN(type, *args, **kwargs):
    if type == 'lstm':
        return nn.LSTM(*args, **kwargs)
    elif type == 'gru':
        return nn.GRU(*args, **kwargs)
    else:
        raise ValueError('rnn_type has to be either lstm or gru')