import math

def loss_metric(input, output, ground_truth, nll):
    return nll / ground_truth[0].size(0)

def perplexity(input, output, ground_truth, nll):
    nll /= ground_truth[0].size(0)
    return math.exp(nll)