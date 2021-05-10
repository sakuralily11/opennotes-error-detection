# Import libraries
import os 
import re
import pickle
import random
import subprocess
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from collections import Counter

def get_tokenized_sentences(notes_df, hadm_id, filter_labs=True, filter_titles=True):
  result_dict = {}

  subset = notes_df.loc[notes_df["HADM_ID"] == hadm_id]

  # Filtering out duplicate / autosave's -- only take the longest
  for cat in subset.CATEGORY.unique(): 
    cat_subset = subset.loc[subset.CATEGORY == cat]
    for time in cat_subset.CHARTTIME.unique():
      cat_time_subset = cat_subset.loc[cat_subset.CHARTTIME == time]
      if len(cat_time_subset) > 1:
        # get indices of first N-1 shortest rows
        idx_to_drop = cat_time_subset.TEXT.apply(lambda x: len(x)).sort_index().index[:-1]
        subset = subset.drop(idx_to_drop) # drop by row index
        
  # for each type of note / daily note for hadm_id
  for i in tqdm(range(len(subset))):
    row = subset.iloc[i]
    note = row["TEXT"]
    row_id = row["ROW_ID"]
    
    # runs command and returns stdout as bytes, convert to utf-8, list of sentences
    sents = subprocess.check_output(f"python mimic-tokenize/heuristic-tokenize.py {note}".split(" "))
    sents = sents.decode("utf-8")
    ind_sentences = sents.split(", \'")
    
    if filter_labs:
        ind_sentences = delete_copied_lab_tables(ind_sentences)
    if filter_titles:
        ind_sentences = remove_titles(ind_sentences)

    # Get time
    if type(row.CHARTTIME)==str:
        time = datetime.strptime(row.CHARTTIME, "%Y-%m-%d %H:%M:%S")
    elif type(row.CHARTDATE)==str:
        time = datetime.strptime(row.CHARTDATE, "%Y-%m-%d")
    else:
        time = None
        
    result_dict[row_id] = (ind_sentences, time, row.CATEGORY)
  
  # result_dict key is the row id in note_df
  # result_dict value is the list of tokenized sentences
  return result_dict

def diff_list(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def delete_copied_lab_tables(ind_sentences):

  # [**yyyy-mm-dd**], 02:10
#   rgx_list = ["[\*\*\d{4}\-\d{1,2}\-\d{1,2}\*\*]", "\d{1,2}\-\d{1,2}"]
#   rgx_list = ["[\*\*[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\*\*] *[0-9]{1,2}-[0-9]{1,2}"]
#   rgx_list = ["[\*\*[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\*\*]   [0-9][0-9]-[0-9][0-9]"]
  rgx_list = ["[\*\*[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\*\*]"]
#   rgx_list = ["[\d{4}\-\d{1,2}\-\d{1,2}][^\S]+\d{1,2}\-\d{1,2}"]

  delete_list = []
  # ind_sentences is list of strings
  for sentence in ind_sentences:
    for rgx_match in rgx_list:
      match = re.search(rgx_match, sentence)
      if match and sentence not in delete_list:
        delete_list.append(sentence)
#         print(f"Sentence: {sentence}")
#         print("Deleted")

  return diff_list(ind_sentences, delete_list)

def remove_titles(sentences):
    """ Omits anything that has ':' in last two entries of the string. 
    e.g. "...Results:"
    """
    return list(filter(lambda x: ':' not in x[-2:], sentences))
