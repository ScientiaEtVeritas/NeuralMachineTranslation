import math
import os
import re
import numpy as np

def loss_metric(nll, target_length, **kwargs):
    return nll / target_length

def perplexity(nll, target_length, **kwargs):
    return math.exp(loss_metric(nll, target_length))

def bleu(real_target_sentence, estimated_target_sentence, **kwargs):
    with open('./Output.txt', 'w') as output_file, open('./Reference.txt', 'w') as reference_file:
        output_file.write(estimated_target_sentence)
        reference_file.write(real_target_sentence)
    value = os.popen('perl ./multi_bleu.pl ./Reference.txt < ./Output.txt').read()
    
    #value format : BLEU = 77.88, 100.0/100.0/100.0/100.0 (BP=0.779, ratio=0.800, hyp_len=4, ref_len=5)
    #extract five BLEU descriptors and return with a tensor to permit of using the same sum, division, append operations which are defined with the other metrics
    print(f"value of BLEU : {value}")
    return np.array([float(x) for x in re.findall(r"[0-9]+[.]?[0-9]*",value)[:5]])
    
