# David Pitt
# Preprocessing step for the MIMIC-III dataset 
# This is the first step in running HORDE


data_dir = '/home/dave/Desktop/College/Math189AC/mimic3_1.4_files/'

import pandas as pd
import csv
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import sys
import pickle
from sklearn import model_selection
import argparse
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix


class EncounterInfo(object):
    def __init__(self, patient_id, encounter_id, encounter_timestamp, expired):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.encounter_timestamp = encounter_timestamp
        self.expired = expired
        self.dx_ids = []
        self.rx_ids = []
        self.labs = {}
        self.physicals = []
        self.treatments = []
        self.concepts = []


def process_patient(infile, encounter_dict, min_length_of_stay=0):
    inff = open(infile, 'r')
    count = 0
    patient_dict = {}
    for line in csv.DictReader(inff):
        if count % 100 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = line['SUBJECT_ID']
        encounter_id = line['HADM_ID']
        encounter_timestamp = line['ADMITTIME']
        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append((encounter_timestamp, encounter_id))
        count += 1
    inff.close()
    print('')
    print(len(patient_dict))
    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples)

    inff = open(infile, 'r')
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = line['SUBJECT_ID']
        encounter_id = line['HADM_ID']
        encounter_timestamp = datetime.strptime(line['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        expired = line['HOSPITAL_EXPIRE_FLAG'] == "1"
        if (datetime.strptime(line['DISCHTIME'], '%Y-%m-%d %H:%M:%S') - encounter_timestamp).days < min_length_of_stay:
            continue

        ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired)
        if encounter_id in encounter_dict:
            print('Duplicate encounter ID!!')
            print(encounter_id)
            sys.exit(0)
        encounter_dict[encounter_id] = ei
        count += 1
    inff.close()
    print('')
    return encounter_dict


def process_diagnosis(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['HADM_ID']
        #dx_id = line['ICD9_CODE'].lower()
        dx_id = line['ICD9_CODE'] # not lowercase for HORDE by necessity
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].dx_ids.append(dx_id)
        count += 1
    inff.close()
    print('')
    print('Diagnosis without Encounter ID: %d' % missing_eid)
    return encounter_dict


def process_treatment(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['HADM_ID']
        treatment_id = line['ICD9_CODE'].lower()
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].treatments.append(treatment_id)
        count += 1
    inff.close()
    print('')
    print('Treatment without Encounter ID: %d' % missing_eid)
    return encounter_dict

def process_notes(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['HADM_ID']
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].concepts = line['TEXT'].split()
        count += 1
    inff.close()
    print('')
    print('Notes without Encounter ID: %d' % missing_eid)
    return encounter_dict

def build_npy(enc_dict,skip_duplicate=False, min_num_codes=1,max_num_codes=50):
    encounter_seq = {}

    items = []
    for e in enc_dict.items():
        enc  = e[1]
        print(enc)
        if enc.patient_id in encounter_seq.keys():
            encounter_seq[enc.patient_id] += 1
        else:
            encounter_seq[enc.patient_id] = 1

        item = [enc.patient_id,encounter_seq[enc.patient_id],enc.dx_ids,enc.concepts]
        items.append(item)
    
    data = np.array(items,dtype=object)
    return data




"""Set <input_path> to where the raw MIMIC CSV files are located.
Set <output_path> to where you want the output files to be.
"""


def main():
    parser = argparse.ArgumentParser(description='File path')
    parser.add_argument('--input_path', type=str, default='.', help='input path of original dataset')
    parser.add_argument('--output_path', type=str, default='.', help='output path of processed dataset')
    args = parser.parse_args()
    #input_path = args.input_path
    input_path = data_dir
    output_path = args.output_path

    admission_dx_file = input_path + '/ADMISSIONS.csv'
    diagnosis_file = input_path + '/DIAGNOSES_ICD.csv'
    treatment_file = input_path + '/PROCEDURES_ICD.csv'
    note_file = input_path + '/NOTEEVENTS_NLP.csv'
    encounter_dict = process_patient(admission_dx_file, {})
    encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
    encounter_dict = process_treatment(treatment_file, encounter_dict)
    encounter_dict = process_notes(note_file, encounter_dict)

    #print(encounter_dict)
    np_array = build_npy(encounter_dict)
    np.save(output_path + 'mimic3_vseq.npy', np_array)
    print('done')


if __name__ == '__main__':
    main()
