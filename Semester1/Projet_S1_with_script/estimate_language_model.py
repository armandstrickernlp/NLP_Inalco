import pprint as pp
from collections import defaultdict
import numpy as np
import argparse

def make_trigrams(text):
    text = text.replace('.', ' .').replace(',', ' ,')
    text_tokenized = text.split()
  
    trigrams = []

    for idx in range(len(text_tokenized) - 2):
        trigrams.append((text_tokenized[idx], text_tokenized[idx+1], text_tokenized[idx+2]))
    
    return trigrams


def make_conditional_probas(text_file):

    proba_table = defaultdict(dict)
    
    with open(text_file, 'r') as f:
        for line in f:
            trigrams = make_trigrams(line)
            for trigram in trigrams:
                bigram = trigram[:2]
                next_word = trigram[2]

                if bigram in proba_table:
                    next_word_counts = proba_table[bigram]
                else:
                    next_word_counts = defaultdict(int)
                next_word_counts[next_word] += 1
                proba_table[bigram] = next_word_counts
    
    for bigram in proba_table:
        next_word_counts = proba_table[bigram]
        next_word_probas = {key: value/sum(next_word_counts.values()) for (key, value) in next_word_counts.items()}
        proba_table[bigram] = next_word_probas        
    
    return proba_table

def sample_from_discrete_distrib(distrib):
    words, probas = zip(*distrib.items())
    probas = np.asarray(probas).astype('float64')/np.sum(probas)
    return np.random.choice(words, p=probas)


def generate(proba_table):
    
    w_i_minus2 = 'BEGIN'
    w_i_minus1 = 'NOW'
    w_i = None
    
    generated_sent = f"{w_i_minus2} {w_i_minus1} "

    while w_i != 'END':
        h = (w_i_minus2, w_i_minus1)
        w_i = sample_from_discrete_distrib(proba_table[h])
        generated_sent += f"{w_i} "
        w_i_minus2, w_i_minus1 = w_i_minus1, w_i

    return "\n"+generated_sent+"\n"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, default='test_reviews.txt', help='text file to train model on')
    args = parser.parse_args()
    
    proba_table = make_conditional_probas(args.text_file)
    print(generate(proba_table))

    # python estimate_language_model.py --text_file wine2.txt (or test_reviews.txt)

