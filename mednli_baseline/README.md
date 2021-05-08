MedNLI Baseline
===============
A simple baseline for Natural Language Inference in clinical domain using the MedNLI dataset.
Includes simplified CBOW and InferSent models from the corresponding paper. 

Installation
------------

1. Run `conda env create -f environment.yml`
2. Run `conda activate mednli`
3. Run a python shell, then execute:
	`>> import nltk`
	`>> nltk.download('punkt')`

Downloading the dataset, word embeddings, and pre-trained models
----------------------------------------------------------------
1. In the `./data` directory:
    1. Create the `./data/cache` directory 
    2. Create the `./data/models` directory
    3. Create the `./data/word_embeddings` directory
2. Download word embeddings (see the table below) and put the `*.pickled` files into the `./data/word_embeddings/` dir (`wget -P data/word_embeddings/ http://mednli.blob.core.windows.net/shared/word_embeddings/mimic.fastText.no_clean.300d.pickled`)
1. Download pre-trained models by running:
   1. `wget -P data/models/ http://m
ednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.mimic.128.sariedpg.pkl`
   2. `wget -P data/models/ http://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.mimic.128.sariedpg.pt`

Using a pre-training model
--------------------------
Run the `predict.py` file with three arguments:
1. Path to the model specification file (`*.pkl`)
1. Input file in the `jsonl` format (see `mli_dev_v1.jsonl`) or the `\t`-separated premise and hypothesis (see [test_input.txt](https://mednli.blob.core.windows.net/shared/test_input.txt) or `data/input_test_small.txt`) 
1. Output file `.csv` to save predicted probabilities of each of the three classes (contradiction, entailment, and neutral)

Notes:
1. The model weights file (`*.pt`) should be located in the same dir as the model specification file (`*.pkl`)
1. In case of the `jsonl` format the sentences are taken from the `sentence1_binary_parse` and `sentence2_binary_parse` fields,
 where the `sentence1` is the premise and `sentence2` is the hypothesis. All other fields are optional

Example command to run the prediction:
```
python predict.py data/models/mednli.infersent.mimic.128.sariedpg.pkl data/input_test_small.txt data/predictions_test.csv
```

Training the model
------------------

Run the `train.py` file. The options are set in the `config.py` file. Command-line interface is coming soon!
By default, the model specification and the model weights are saved in the `./data/models` dir.

Training the feature based system
------------------

To run a traditional feature based system, run the `train_feature_based.py` file. 
This system achieves 0.523 accuracy on the dev set using a gradient boosting classifier 
with features based on word overlaps, tf-idf similarities, word embeddings similarities, and blue scores.


# Reference

Romanov, A., & Shivade, C. (2018). Lessons from Natural Language Inference in the Clinical Domain. arXiv preprint arXiv:1808.06752.  
https://arxiv.org/abs/1808.06752


```
@article{romanov2018lessons,
	title = {Lessons from Natural Language Inference in the Clinical Domain},
	url = {http://arxiv.org/abs/1808.06752},
	abstract = {State of the art models using deep neural networks have become very good in learning an accurate mapping from inputs to outputs. However, they still lack generalization capabilities in conditions that differ from the ones encountered during training. This is even more challenging in specialized, and knowledge intensive domains, where training data is limited. To address this gap, we introduce {MedNLI} - a dataset annotated by doctors, performing a natural language inference task ({NLI}), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. {SNLI}) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.},
	journaltitle = {{arXiv}:1808.06752 [cs]},
	author = {Romanov, Alexey and Shivade, Chaitanya},
	urldate = {2018-08-27},
	date = {2018-08-21},
	eprinttype = {arxiv},
	eprint = {1808.06752},
}
