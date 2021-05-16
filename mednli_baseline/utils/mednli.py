import json

from nltk import word_tokenize


def get_tokens(sentence_binary_parse):
    sentence = sentence_binary_parse \
        .replace('(', ' ').replace(')', ' ') \
        .replace('-LRB-', '(').replace('-RRB-', ')') \
        .replace('-LSB-', '[').replace('-RSB-', ']')

    tokens = sentence.split()

    return tokens


def read_mednli(filename):
    data = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            example = line.split('\t')
            if len(example) != 3:
                continue

            premise = get_tokens(example[0])
            hypothesis = get_tokens(example[1])
            label = num_to_label(example[2].rstrip())

            data.append((premise, hypothesis, label))

    print(f'MedNLI file loaded: {filename}, {len(data)} examples')
    return data

def num_to_label(l):
    if l == "0":
        return "entailment"
    else:
        return "contradiction"


def read_sentences(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        lines = [l.split('\t') for l in f.readlines()]

    # TODO: if using training or predicting with labeled data, use line 47; if predicting without labels, use line 48
    #input_data = [(word_tokenize(l[0]), word_tokenize(l[1]), num_to_label(l[2].rstrip())) for l in lines if len(l) == 3]
    #input_data = [(word_tokenize(l[0]), word_tokenize(l[1]), None) for l in lines if len(l) == 2]
    return input_data


def load_mednli(cfg):
    # TODO: change these file paths to where you saved the dataset files
    prefix = 'generated/'
    filenames = [
        prefix+'train.txt',
        prefix+'dev.txt',
        prefix+'test.txt',
    ]
    filenames = [cfg.mednli_dir.joinpath(f) for f in filenames]

    mednli_train, mednli_dev, mednli_test = [read_mednli(f) for f in filenames]

    return mednli_train, mednli_dev, mednli_test
