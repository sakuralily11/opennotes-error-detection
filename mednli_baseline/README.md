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
    4. Create the `.data/mednli` directory
2. Download word embeddings (see the table below) and put the `*.pickled` files into the `./data/word_embeddings/` dir (`wget -P data/word_embeddings/ http://mednli.blob.core.windows.net/shared/word_embeddings/mimic.fastText.no_clean.300d.pickled`)
3. See the main opennotes directory instructions for downloading datasets. They should be put in the `data/mednli` directory.

Put Data into Compatible Form
-----------------------------

In order to run `train.py`, the dataset must be split into three batches: train, dev, and test. Each batch should be in a `.txt` file, in which each line contains a sentence pair and the label(1 for contradiction, 0 otherwise) separated by tabs. To get the data in a form, see `parse.ipynb`, which has functions to transform csv files into the format needed. 

Configuring Model Type
----------------------

To train an InferSent or bag-of-words(Simple) model, the model type must be changed in `config.py`. Set this according to the desired model type.

Training the Model
---------------------

Once data is in the proper form and the desired model type is selected, we can trin the model. In the files `train.py` and `utils/mednli.py` there are TODOs with instructions to set file paths for data and models. Set these according to where data is saved and which dataset is being trained. Once done, run the `train.py` file. By default, the model specification and the model weights are saved in the `./data/models` dir.

Evaluating the Model
--------------------

To evaluate the model, run `predict.py` with the following command `python predict.py data/models/[saved_model_path].pkl data/mednli/[evaluation_data_txt] data/[filename].csv`. The evaluation data file must be in the same form as training data, but may exclude the labels. Follow TODOs in `predict.py` based on whether using labled or unlabeled data.


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
