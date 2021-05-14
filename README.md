# OpenNotes: ConflictClassifier

ConflictClassifier is an end-to-end rule-based pipeline that: 

1) takes unstructured notes and structured table data as input and extracts candidate sentence pairs for comparison, and 

2) classifies whether the sentence pairs contradict.

We integrate domain-specific knowledge bases, such as UMLS and RxNorm ontologies, for candidate sentence extraction, and construct domain-specific features for rule based classification. We use the [MedNLI](https://physionet.org/content/mednli/1.0.0/) dataset and construct a dataset from [MIMIC-III](https://mimic.physionet.org/). We also compare against non-rule based baselines from [Romanov and Shivade](https://arxiv.org/abs/1808.06752), which give insights into simple, interpretable models versus more complex, black-box models, especially in the context of limited data. 

## Installation

Using `conda`, set up a virtual environment with `environment.yml` and activate, e.g. 

```
# creates virtual environment and installs dependencies
conda env create -f environment.yml

# activates opennotes venv
conda activate opennotes
```

## Loading MedNLI and MIMIC-III

After getting access to MedNLI and MIMIC-III data through PhysioNet, use `gsutil` command to copy MIMIC-III tables from Google Cloud Storage Bucket to your local or virtual machine, and download MedNLI directly from [here](https://physionet.org/content/mednli/1.0.0/).

```
# authenticate, this should direct you to a link; click on account linked to PhysioNet and copy+paste authentication code to verify credentials
gcloud auth login 

# copy files (mimic-iii)
gsutil cp gs://mimiciii-1.4.physionet.org/NOTEEVENTS.csv.gz .
gsutil cp gs://mimiciii-1.4.physionet.org/PRESCRIPTIONS.csv.gz .
gsutil cp gs://mimiciii-1.4.physionet.org/LABEVENTS.csv.gz .
gsutil cp gs://mimiciii-1.4.physionet.org/D_LABITEMS.csv.gz .
```

## Data Processing

todo, @yuria

talk about data structures file too and processing

### MIMIC-III

todo, @yuria

`Generating Contradictions - Yuria.ipynb` has all the data loading and processing code for MIMIC-III (unlabeled) and generated MIMIC-III data. 

### MedNLI 

todo, @yuria

`MedNLI Data Processing.ipynb` has all the data loading and processing code for MedNLI data. 

## Classification

### Baseline

todo, @diana

instructions on how to reproduce results - e.g. run this code to reproduce table X/experiment A results

### Rule-Based 

todo, @sharon

instructions on how to reproduce results - e.g. run this code to reproduce table X/experiment A results
