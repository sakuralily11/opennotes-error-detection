# Import libraries
import os 
import random
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from collections import Counter

from data_helper import get_tokenized_sentences
from sentence_rep import SentRep

# Load NOTEEVENTS table
data = pd.read_csv('NOTEEVENTS.csv.gz', compression='gzip', error_bad_lines=False)

# Load HADM ID's with consecutive physician notes
if os.path.exists("hadm_ids.pkl"):
    with open("hadm_ids.pkl", "rb") as f:
        hadm_ids = pickle.load(f)
else:
    hadm_ids = []
    for hadm_id in tqdm(data.HADM_ID.unique()):
        hadm_data = data.loc[data.HADM_ID == hadm_id]
        hadm_phys_notes = hadm_data.loc[hadm_data.CATEGORY == "Physician "]

        if len(hadm_phys_notes) > 1:
            hadm_ids.append(hadm_id)

    with open("hadm_ids.pkl", "wb") as f:
        pickle.dump(hadm_ids, f)

hadm_id = hadm_ids[1]
print(f"HADM ID={hadm_id}")

# get dict of list of sentences for each note
sents_dict = get_tokenized_sentences(data, hadm_id)

# get only sentences from physician notes
all_sent_strs  = []  # sentences
all_sent_times = []  # corresponding times
for row_id, (row_sents, row_time, row_category) in sents_dict.items():
#     if 'physician' in row_category.lower() or 'nursing' in row_category.lower():
    if 'physician' in row_category.lower():
        all_sent_strs.extend(row_sents)
        all_sent_times.extend([row_time for _ in range(len(row_sents))])
        
print(f"Total number of sentences from physician notes: {len(all_sent_strs)}")


import spacy
import scispacy

from pprint import pprint
from collections import OrderedDict

from spacy import displacy
# from scispacy.abbreviation import AbbreviationDetector # UMLS already contains abbrev. detect
from scispacy.umls_linking import UmlsEntityLinker

# 0. Initialize conflict category and semantic type mapping
SEMANTIC_TYPES = ['T047', 'T121', 'T023', 'T061', 'T060', 'T059', 'T034', 'T184', 'T200']
SEMANTIC_NAMES = ['Disease or Syndrome', 'Pharmacologic Substance', 'Body Part, Organ, or Organ Component', \
                  'Therapeutic or Preventive Procedure', 'Diagnostic Procedure', 'Laboratory Procedure', \
                  'Laboratory or Test Result', 'Sign or Symptom', 'Clinical Drug']
SEMANTIC_TYPE_TO_NAME = dict(zip(SEMANTIC_TYPES, SEMANTIC_NAMES))

CONFLICT_TO_SEMANTIC_TYPE = {
    "diagnosis": {'T047', 'T060'},
    "med_history_symptom": {'T184'},
    "med_history_operation": {'T061'},
    "med_history_other": set(SEMANTIC_TYPES),
    "med_allergy": {'T200', 'T121'},
    "test_results": {'T059', 'T034'}
}

# 1. Set up model
sci_nlp = spacy.load("en_core_sci_sm")

# 2. Get entity linker
# abbreviation_pipe = AbbreviationDetector(nlp) # automatically included with UMLS linker
# nlp.add_pipe(abbreviation_pipe)
linker = UmlsEntityLinker(k=10,                          # number of nearest neighbors to look up from
                          threshold=0.7,                 # confidence threshold to be added as candidate
                          max_entities_per_mention=1,    # number of entities returned per concept (todo: tune)
                          filter_for_definitions=False,  # no definition is OK
                          resolve_abbreviations=True)    # resolve abbreviations before linking
sci_nlp.add_pipe(linker)

# 3. Process sentences in a note
all_sreps = []
all_srep_canon_names = []
all_srep_times = []
for i, sent in tqdm(enumerate(all_sent_strs)):
    try: # some sentences run into errors it seems.. todo: look into
        srep = SentRep(sent, sci_nlp, linker,
                       filter_map=SEMANTIC_TYPE_TO_NAME,
                       conflict_map=CONFLICT_TO_SEMANTIC_TYPE)
        if len(srep.canonical_names) > 0:
            all_sreps.append(srep)
            all_srep_canon_names.append(list(srep.canonical_names))
            all_srep_times.append(all_sent_times[i])
    except:
        continue
print(f"Total number of processed sentences: {len(all_sreps)}")
