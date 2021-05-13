# Import libraries
import os
import re
import random
import pickle
import subprocess
import numpy as np
import pandas as pd
import datetime as dt

from tqdm import tqdm
from datetime import datetime
from collections import Counter

"""
Constants: see Jupyter notebook for details
"""
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


""" Part I: Patient """

class Patient(object):
    def __init__(self, hadm_id, notes_df, prescription_df, lab_df, d_lab_df, \
                 med7_nlp, umls_nlp, rxnorm_nlp, umls_linker, rxnorm_linker, \
                 physician_only=True):
        """ Patient representation
        
        med7_nlp:      spacy model from Med7
        umls_nlp:      spacy model with UMLS entity linker
        rxnorm_nlp:    spacy model with RxNorm entity linker
        umls_linker:   entity linker for UMLS, should already be linked to umls_nlp
        rxnorm_linker: entity linker for RxNorm, should already be linked to rxnorm_nlp
        """
        self.hadm_id = hadm_id
        self.physician_only = physician_only
        
        # this patient's data
        self.notes_df = self.filter_notes(notes_df.loc[notes_df['HADM_ID'] == hadm_id])
        self.prescription_df  = prescription_df.loc[prescription_df['HADM_ID'] == hadm_id]
        self.lab_df   = lab_df.loc[lab_df['HADM_ID'] == hadm_id]
        
        self.d_lab_df = d_lab_df # lab ditems df
        
        # spaCy models & entity linkers
        self.med7   = med7_nlp
        self.umls   = umls_nlp
        self.rxnorm = rxnorm_nlp
        self.umls_linker   = umls_linker
        self.rxnorm_linker = rxnorm_linker
        
        # Process notes, prescriptions, and labs
        self.process_notes()
        self.process_prescriptions()
        self.process_labs()                # note: comment out if removing labs
        
        # Final. Process all data (notes, prescriptions, labs), map by date
        self.process_by_date()
        
    def filter_notes(self, pat_notes_df):
        if self.physician_only: pat_notes_df = self._filter_physician(pat_notes_df)
        pat_notes_df = self._filter_duplicates(pat_notes_df)
        
        return pat_notes_df
    
    def process_by_date(self):
#         start, end, (notes_dates, prescriptions_dates, lab_dates) = self._get_start_end_dt()
        start, end, all_dates = self._get_start_end_dt()
        delta = dt.timedelta(days=1)
        current = start
        dailydata = {}
        while current <= end:
            current_dailydata = self._get_current_dailydata(current, all_dates)
            if len(current_dailydata) > 0: 
                dailydata[current] = current_dailydata
            current += delta
        self.dailydata = dailydata
    
    def process_notes(self):
        notes = []
        for row_id in self.notes_df.ROW_ID:
            note = Note(self, row_id)
            notes.append(note)
        self.notes = notes
        
    def process_prescriptions(self):
        start, end = self._get_prescription_start_end_dt()  # get start/end dates
        self._process_prescription_sents()                  # get prescription info in sentence form

        # for each date, get all the prescriptions given and construct PrescriptionOrders
        delta = dt.timedelta(days=1)
        current = start
        prescriptions = []
        while current <= end:
            current_prescription_df = self.prescription_df.apply(lambda x: x.START_DT <= current and x.END_DT >= current, axis=1)
            if current_prescription_df.sum() > 0: # if there is at least 1
                prescription_order = PrescriptionOrders(self, current_prescription_df, current)
                prescriptions.append(prescription_order)

            current += delta # go to next date
        self.prescriptions = prescriptions
        
    def process_labs(self):
        start, end = self._get_lab_start_end_dt() # get start/end dates
        self._process_lab_sents()                 # get lab info in sentence form
        
        # for each date, get all labs done and construct LabResults
        delta = dt.timedelta(days=1)
        current = start
        labs = []
        while current <= end:
            current_lab_df = self.lab_df.apply(lambda x: x.DT == current, axis=1)
            if current_lab_df.sum() > 0: # if there is at least 1
                lab_result = LabResults(self, current_lab_df, current)
                labs.append(lab_result)

            current += delta # go to next date
        self.labs = labs
    
    def _get_prescription_start_end_dt(self):
        # get datetimes for start and end dates
        start_dt = self.prescription_df.STARTDATE.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
        end_dt   = self.prescription_df.ENDDATE.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())

        self.prescription_df[['START_DT']] = start_dt
        self.prescription_df[['END_DT']]   = end_dt

        # get earliest and latest dates
        start = min(start_dt)
        end   = max(end_dt)
        
        return start, end
    
    def _get_lab_start_end_dt(self):
        # get datetimes for start and end dates
        time_dt = self.lab_df.CHARTTIME.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
        
        self.lab_df[['DT']] = time_dt
        
        # get earliest and latest dates
        start = min(time_dt)
        end   = max(time_dt)
        
        return start, end
    
    def _process_prescription_sents(self):
        # process sentence for each prescription
        # get_prescription_sent = lambda row: f"Patient was prescribed {row.DRUG.item()} {row.PROD_STRENGTH.item()} {row.ROUTE.item()} of total {row.DOSE_VAL_RX.item()} {row.DOSE_UNIT_RX.item()}"
        get_prescription_sent = lambda row: f"Patient was prescribed {row.DRUG} {row.PROD_STRENGTH} {row.ROUTE} of total {row.DOSE_VAL_RX} {row.DOSE_UNIT_RX}"
        prescription_sents = self.prescription_df.apply(get_prescription_sent, axis=1)
        self.prescription_df[['Sentence']] = prescription_sents
        
    def _process_lab_sents(self):
        # process lab for each lab
        def _get_lab(row):
            """ Gets name of lab """
            return self.d_lab_df.loc[self.d_lab_df.ITEMID == row.ITEMID].LABEL.item()

        def _get_flag(row):
            """ Gets normal/abnormal flag info """
            if row.NULL_FLAG: # no flag info if no flag
                return ""
            return f" , which is {row.FLAG}"
        
        # null flag checking
        self.lab_df[['NULL_FLAG']] = self.lab_df.FLAG.isnull() # True if null, False otherwise
        
        get_lab_sent = lambda row: f"Patient's {_get_lab(row)} lab came back {row.VALUE} {row.VALUEUOM}{_get_flag(row)}."
        lab_sents = self.lab_df.apply(get_lab_sent, axis=1)
        self.lab_df[['Sentence']] = lab_sents
    
    def _filter_physician(self, pat_notes_df):
        # Filter for only physician notes
        return pat_notes_df.loc[pat_notes_df.CATEGORY == "Physician "]
        
    def _filter_duplicates(self, pat_notes_df):
        # Filtering out duplicate / autosave's -- only take the longest
        for cat in pat_notes_df.CATEGORY.unique(): 
            cat_notes_df = pat_notes_df.loc[pat_notes_df.CATEGORY == cat]
            for time in cat_notes_df.CHARTTIME.unique():
                time_notes_df = cat_notes_df.loc[cat_notes_df.CHARTTIME == time]
                if len(time_notes_df) > 1:
                    # get indices of first N-1 shortest rows
                    idx_to_drop = time_notes_df.TEXT.apply(lambda x: len(x)).sort_index().index[:-1]
                    pat_notes_df = pat_notes_df.drop(idx_to_drop) # drop by row index
                    
        return pat_notes_df

    def _get_start_end_dt(self, return_all_dates=True):
        """ Gets start and end datetimes across all data. todo: add labs"""
        notes_dates         = list(map(lambda x: x.time.date(), self.notes))
        prescriptions_dates = list(map(lambda x: x.date,        self.prescriptions))
        lab_dates           = list(map(lambda x: x.date,        self.labs))
        
        start = min(notes_dates + prescriptions_dates + lab_dates)
        end   = max(notes_dates + prescriptions_dates + lab_dates)
        
        return start, end, (notes_dates, prescriptions_dates, lab_dates)

    def _get_current_items(self, items, dates, current):
        """ Get items for current date """
        items_and_dates = zip(items, dates)
        current_items_and_dates = filter(lambda x: x[1] == current, items_and_dates)
        
        try:
            current_items, current_dates = list(zip(*current_items_and_dates))
            return list(current_items)
        except ValueError: # if current items don't exist
            return []
        
    def _get_current_dailydata(self, current, all_dates):
        """ Gets DailyData instances for current date.
        
        all_dates: iterable of dates, corresponding to self.[DailyData list]
        """
        notes_dates, prescriptions_dates, lab_dates = all_dates
        
        current_notes         = self._get_current_items(self.notes,         notes_dates,         current)
        current_prescriptions = self._get_current_items(self.prescriptions, prescriptions_dates, current)
        current_labs          = self._get_current_items(self.labs,          lab_dates,           current)
        
        current_data = current_notes + current_prescriptions + current_labs
        
        return current_data

    
""" Part II: Daily Data """

class DailyData(object):
    """ Collection of data from same day. e.g. clinical notes, lab tests, prescription orders. """
    def __init__(self, patient):
        self.patient  = patient   # patient this data is for
    
    def __getitem__(self, idx):
        return self.datas[idx]

    @property
    def hadm_id(self):
        return self.patient.hadm_id
    
    @property
    def datas_txts(self):
        return list(map(lambda x: x.txt, self.datas))
    
    @property
    def datas_features(self):
        return list(map(lambda x: x.features, self.datas))
    
    @property
    def datas_semantic_types(self):
        return list(map(lambda x: x.semantic_types, self.datas))

    @property
    def datas_semantic_names(self):
        return list(map(lambda x: x.semantic_names, self.datas))

    @property
    def med7(self):
        return self.patient.med7
    
    @property
    def umls(self):
        return self.patient.umls
    
    @property
    def rxnorm(self):
        return self.patient.rxnorm
    
    @property
    def umls_linker(self):
        return self.patient.umls_linker

    @property
    def rxnorm_linker(self):
        return self.patient.rxnorm_linker

class Note(DailyData):
    def __init__(self, patient, row_id):
        super(Note, self).__init__(patient)
        
        self.row_id   = row_id
        self.note_row = patient.notes_df.loc[patient.notes_df.ROW_ID == row_id]   # df row for this note
        self.txt      = self.note_row.TEXT.item()                                       # note in string format
        self.cat      = self.note_row.CATEGORY.item()                                   # note category
        
        # Get datetime
        if type(self.note_row.CHARTTIME.item()) == str:
            self.time = datetime.strptime(self.note_row.CHARTTIME.item(), "%Y-%m-%d %H:%M:%S")
        elif type(self.note_row.CHARTDATE.item()) == str:
            self.time = datetime.strptime(self.note_row.CHARTDATE.item(), "%Y-%m-%d")
        else:
            self.time = None
            
        # Tokenize note
#         sents = !python mimic-tokenize/heuristic-tokenize.py "{self.txt}"
#         sentences = sents[0].split(", \'")
        # For python script: runs command and returns stdout as bytes, convert to utf-8, list of sentences
        sents = subprocess.check_output(f"python mimic-tokenize/heuristic-tokenize.py {self.txt}".split(" "))
        sents = sents.decode("utf-8")
        sentences = sents.split(", \'")

        # Remove lab tables, remove titles
        sentences = self._delete_copied_lab_tables(sentences)
        sentences = self._remove_titles(sentences)
        
        self.sentences = sentences # todo: process each sentence
        
        # Process each sentence
        self.process_sentences(sentences)

    def process_sentences(self, sentences):
        """ Processes each sentence.
        
        Args:
            sentences: list of strings, where strings are tokenized sentences
        """
        sentence_reps = []
        for idx, sent in enumerate(sentences):
            sent_rep = Sentence(self, idx,
                                filter_map=SEMANTIC_TYPE_TO_NAME,
                                conflict_map=CONFLICT_TO_SEMANTIC_TYPE)
            sentence_reps.append(sent_rep)
            
        self.datas = sentence_reps
        
    def _diff_list(self, li1, li2):
        return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

    def _delete_copied_lab_tables(self, ind_sentences):
        # [**yyyy-mm-dd**], 02:10
#         rgx_list = ["[\*\*\d{4}\-\d{1,2}\-\d{1,2}\*\*]", "\d{1,2}\-\d{1,2}"]
#         rgx_list = ["[\*\*[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\*\*] *[0-9]{1,2}-[0-9]{1,2}"]
#         rgx_list = ["[\*\*[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\*\*]   [0-9][0-9]-[0-9][0-9]"]
        rgx_list = ["[\*\*[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\*\*]"]
#         rgx_list = ["[\d{4}\-\d{1,2}\-\d{1,2}][^\S]+\d{1,2}\-\d{1,2}"]
        
        delete_list = []
        # ind_sentences is list of strings
        for sentence in ind_sentences:
            for rgx_match in rgx_list:
                match = re.search(rgx_match, sentence)
                if match and sentence not in delete_list:
                    delete_list.append(sentence)
        return self._diff_list(ind_sentences, delete_list)
    
    def _remove_titles(self, sentences):
        """ Omits anything that has ':' in last two entries of the string. 
        e.g. "...Results:"
        """
        return list(filter(lambda x: ':' not in x[-2:], sentences))
        
class PrescriptionOrders(DailyData):
    def __init__(self, patient, daily_bools, date):
        """ Patient instance and boolean Series for selecting daily rows. """
        super(PrescriptionOrders, self).__init__(patient)
        self.date = date
        self.time = datetime.combine(date, datetime.min.time())
        
        self.prescription_df = self.patient.prescription_df[daily_bools]  # order dataframe
        self.sentences       = self.prescription_df.Sentence.values       # array of orders in sentence form
        
        # Process each prescription
        prescription_data = []
        for idx, _ in enumerate(self.prescription_df.iterrows()):
            prescript_rep = Prescription(self, idx,
                                         filter_map=SEMANTIC_TYPE_TO_NAME,
                                         conflict_map=CONFLICT_TO_SEMANTIC_TYPE)
            prescription_data.append(prescript_rep)
        
        self.datas = prescription_data

class LabResults(DailyData):
    def __init__(self, patient, daily_bools, date):
        """ Patient instance and boolean Series for selecting daily rows. """
        super(LabResults, self).__init__(patient)
        self.date = date
        self.time = datetime.combine(date, datetime.min.time())
        
        self.lab_df    = self.patient.lab_df[daily_bools]  # results dataframe
        self.sentences = self.lab_df.Sentence.values       # array of results in sentence form
        
        # Process each lab
        lab_data = []
        for idx, _ in enumerate(self.lab_df.iterrows()):
            lab_rep = Lab(self, idx,
                          filter_map=SEMANTIC_TYPE_TO_NAME,
                          conflict_map=CONFLICT_TO_SEMANTIC_TYPE)
            lab_data.append(lab_rep)
        
        self.datas = lab_data
        
""" Part III: Data """

class Data(object):
    def __init__(self, dailydata, txt, filter_map=None, conflict_map=None):
        self.dailydata = dailydata
        self.txt       = txt
        self.time      = dailydata.time
        
        self.umls_cui_map   = dailydata.umls_linker.umls.cui_to_entity # maps CUI to UMLS knowledge base
        self.rxnorm_cui_map = dailydata.rxnorm_linker.kb.cui_to_entity # maps CUI to RxNorm knowledge base
        self.filter_map   = filter_map
        self.conflict_map = conflict_map
        self.is_filter    = (filter_map is not None)
        self.is_conflict  = (conflict_map is not None)
        
        self.umls_doc    = dailydata.umls(self.txt)
        self.rxnorm_doc  = dailydata.rxnorm(self.txt)
        self.med7_doc    = dailydata.med7(self.txt)
        
        self.semantic_types = []
        self.semantic_names = []  # names of categories of entities
        
        self.umls_concepts   = [] # names of types of entities (UMLS)
        self.get_umls_info()
        
        self.rxnorm_concepts = [] # names of types of entities (RxNorm)
        self.get_rxnorm_info()
        
        self.semantic_types  = set(self.semantic_types)
        self.semantic_names  = set(self.semantic_names)
        self.umls_concepts   = set(self.umls_concepts)
        self.rxnorm_concepts = set(self.rxnorm_concepts)
        
        self.med7_entities = []   # list of tuples with (entity word, entity label), e.g. (aspirin, drug)
        self.get_med7_info()
        
    @property
    def features(self):
        """ Returns canonical names of extracted concepts. Used to get cosine similarities. """
        return self.umls_concepts | self.rxnorm_concepts
    
    def get_med7_info(self):
        # list of tuples with (entity word, entity label), e.g. (aspirin, drug)
        self.med7_entities = [(ent.text, ent.label_) for ent in self.med7_doc.ents]
        
    def get_umls_info(self):
        for ent in self.umls_doc.ents: # extract info (umls) for each entity
            # todo: look into this bug, ent._.umls_ents sometimes empty list
            try:
                cui, _ = ent._.umls_ents[0] # assuming `max_entites_per_mention=1` for now
            except IndexError:
                continue
            cui_info = self.umls_cui_map[cui]
            
            if not self.is_filter:
                ent_valid_type_list = [True for _ in cui_info.types] # add everything if no filter
            else: 
                ent_valid_type_list = [t in self.filter_map for t in cui_info.types]
            ent_valid_type = any(ent_valid_type_list) # checks if entity is a valid type
            
            if ent_valid_type: # only add to list if we're not filtering of it's valid
                self.umls_concepts.append(cui_info.canonical_name)
                for (stype, keep) in zip(cui_info.types, ent_valid_type_list):
                    if keep:
                        self.semantic_types.append(stype)
                        self.semantic_names.append(self.filter_map[stype])
                
    def get_rxnorm_info(self):
        for ent in self.rxnorm_doc.ents: # extract info for each rxnorm entity
            try:
                cui, _ = ent._.kb_ents[0] # assuming `max_entites_per_mention=1` for now
            except IndexError:
                continue 
            cui_info = self.rxnorm_cui_map[cui]

            if not self.is_filter:
                ent_valid_type_list = [True for _ in cui_info.types] # add everything if no filter
            else: 
                ent_valid_type_list = [t in self.filter_map for t in cui_info.types]
            ent_valid_type = any(ent_valid_type_list) # checks if entity is a valid type
            
            if ent_valid_type: # only add to list if we're not filtering of it's valid
                self.rxnorm_concepts.append(cui_info.canonical_name)
                for (stype, keep) in zip(cui_info.types, ent_valid_type_list):
                    if keep:
                        self.semantic_types.append(stype)
                        self.semantic_names.append(self.filter_map[stype])
            
    def is_ctype(self, ctype):
        """ Given a conflict type (e.g. "diagnosis"),
            returns True if this sentence falls into that category, False otherwise.
            Returns None if conflict_map is undefined.
        """
        if self.is_conflict: 
            ctype_stypes = self.conflict_map[ctype] # get list of semantic types for this conflict
            return any([stype in ctype_stypes for stype in self.semantic_types])
        return None
    
class Sentence(Data):
    def __init__(self, note, sentence_idx, filter_map=None, conflict_map=None, sentence=None):
        """
        Extracts important information and stores them as attributes. 
        """
        if sentence_idx is None: txt = sentence
        else:                    txt = note.sentences[sentence_idx]
        self.sentence_idx = sentence_idx
        self.type         = "sentence"

        super(Sentence, self).__init__(note, txt, filter_map, conflict_map)
        
    def update_text(self, new_text, run_reprocessing=False):
        """ Updates the text of the sentence. 
        
        Args:
            new_text: new text of sentence
            run_reprocessing: reprocesses features for sentence
        """
        # Since we want to update note.sentences, get the index of this sentence
        # then, update the sentence.
        self.dailydata.sentences[self.sentence_idx] = new_text # = something new
        
        # Once we make all of our updates, re-run the processing from note.sentences
        if run_reprocessing:
            self.dailydata.process_sentences(self.dailydata.sentences)            
        
class Prescription(Data):
    def __init__(self, prescription_order, prescription_idx, filter_map=None, conflict_map=None):
        txt = prescription_order.sentences[prescription_idx]
        self.prescription_idx = prescription_idx
        self.type             = "prescription"
        
        super(Prescription, self).__init__(prescription_order, txt, filter_map, conflict_map)
        
class Lab(Data):
    def __init__(self, lab_result, lab_idx, filter_map=None, conflict_map=None):
        txt = lab_result.sentences[lab_idx]
        self.lab_idx = lab_idx
        self.type    = "lab"
        
        super(Lab, self).__init__(lab_result, txt, filter_map, conflict_map)
        