# opennotes-error-detection

set up virtual environment with `environment.yml`

`Tokens to concepts.ipynb` is the main notebook for now


## How to get data from Google Cloud Storage Bucket to VM

run `gcloud auth login` to go through the authentication process. this should direct you to a link: click on the account linked to physionet and copy+paste authentication code to verify credentials.

once you do that, you should be able to run `gsutil cp gs://mimiciii-1.4.physionet.org/[table of your choice].csv.gz .` to copy over the data to the virtual machine

## Some notes for now

`Generating Contradictions - Yuria.ipynb` has all the data loading and processing code for MIMIC-III (unlabeled) and generated MIMIC-III data. 

`MedNLI Data Processing.ipynb` has all the data loading and processing code for MedNLI data. 


