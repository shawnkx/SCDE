import json
import pickle
import sys
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse


def single_prev_next(in_file, out_file, data_type):
	with open(in_file) as f:
		datas = json.load(f)
	quesiton_idx = 0
	examples = []
	total = 0
	answers = []
	with open(out_file, 'w') as of:
		for idx, data in enumerate(datas):
			eid = data['eid']

			passage = data['passage']
			candidates = data['candidates']
			answers = data['answer_sequence']
			num_blanks = data['number_of_blanks']
			num_candidates = data['candidate_length']
			blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
			
			for b_idx, a_idx in zip(blank_indexes, answers):
				if data_type == 'train':
					if b_idx + 1 in blank_indexes or b_idx - 1 in blank_indexes:
						total += 1
						continue
				for c_idx, candi in enumerate(candidates):

					if c_idx == a_idx[1]:
						label = '0'
					else:
						label = '1'

					of.write('\t'.join((str(quesiton_idx), passage[b_idx-1], candi, label)) + '\n')
					next_id = b_idx + 1
					while next_id < len(passage) and passage[next_id] == ".":
						next_id += 1
					if next_id < len(passage):
						of.write('\t'.join((str(quesiton_idx), candi, passage[next_id], label)) + '\n')
					else:
						of.write('\t'.join((str(quesiton_idx), candi, " ",  label)) + '\n')
					
					quesiton_idx += 1

def all_prev_next(in_file, out_file, data_type):
	with open(in_file) as f:
		datas = json.load(f)
	quesiton_idx = 0
	examples = []
	total = 0
	answers = []
	with open(out_file, 'w') as of:
		for idx, data in enumerate(datas):
			eid = data['eid']

			passage = data['passage']
			candidates = data['candidates']
			answers = data['answer_sequence']
			num_blanks = data['number_of_blanks']
			num_candidates = data['candidate_length']
			blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(len(answers))]
			for b_idx, a_idx in zip(blank_indexes, answers):
				if data_type == 'train':
					if b_idx + 1 in blank_indexes or b_idx - 1 in blank_indexes:
						total += 1
						continue
				for c_idx, candi in enumerate(candidates):
					if c_idx == a_idx[1]:
						label = '0'
					else:
						label = '1'
					# all prev
					all_previous =  ' '.join(passage[:b_idx]).strip().split()
					num_tokens = 20000000
					len_candi = len(candi.strip().split())
					if len(all_previous) + len_candi > num_tokens:
						remove_token = len(all_previous) + len_candi - num_tokens
						all_previous = all_previous[remove_token:]
					of.write('\t'.join((str(quesiton_idx), ' '.join(all_previous), candi, label)) + '\n')
					# all next
					next_id = b_idx + 1
					next_pass = ' '.join(passage[next_id:]).strip().split()
					if len(next_pass) + len_candi > num_tokens:
						remove_token = len(all_previous) + len_candi - num_tokens
						next_pass = next_pass[:-remove_token]
					of.write('\t'.join((str(quesiton_idx), candi, ' '.join(next_pass),  label)) + '\n')

					
					quesiton_idx += 1



def all_next(in_file, out_file, data_type):
	with open(in_file) as f:
		datas = json.load(f)
	quesiton_idx = 0
	examples = []
	total = 0
	answers = []
	with open(out_file, 'w') as of:
		for idx, data in enumerate(datas):
			eid = data['eid']

			passage = data['passage']
			candidates = data['candidates']
			answers = data['answer_sequence']
			num_blanks = data['number_of_blanks']
			num_candidates = data['candidate_length']
			blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
			
			for b_idx, a_idx in zip(blank_indexes, answers):
				if data_type == 'train':
					if b_idx + 1 in blank_indexes:
						total += 1
						continue
				for c_idx, candi in enumerate(candidates):

					if c_idx == a_idx[1]:
						label = '0'
					else:
						label = '1'
					next_id = b_idx + 1
					next_pass = ' '.join(passage[next_id:])
					of.write('\t'.join((str(quesiton_idx), candi, next_pass,  label)) + '\n')
					quesiton_idx += 1
def single_next(in_file, out_file, data_type):
	with open(in_file) as f:
		datas = json.load(f)
	quesiton_idx = 0
	examples = []
	total = 0
	answers = []
	with open(out_file, 'w') as of:
		for idx, data in enumerate(datas):
			eid = data['eid']
			passage = data['passage']
			candidates = data['candidates']
			answers = data['answer_sequence']
			num_blanks = data['number_of_blanks']
			num_candidates = data['candidate_length']
			blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
			for b_idx, a_idx in zip(blank_indexes, answers):
				if data_type == 'train':
					if b_idx + 1 in blank_indexes:
						total += 1
						continue
				for c_idx, candi in enumerate(candidates):
					if c_idx == a_idx[1]:
						label = '0'
					else:
						label = '1'
					next_id = b_idx + 1
					while next_id < len(passage) and passage[next_id] == ".":
						next_id += 1
					if next_id < len(passage):
						of.write('\t'.join((str(quesiton_idx), candi, passage[next_id], label)) + '\n')
					else:
						of.write('\t'.join((str(quesiton_idx), candi, " ", label)) + '\n')
					quesiton_idx += 1


def all_previous(in_file, out_file, data_type):
	with open(in_file) as f:
		datas = json.load(f)
	quesiton_idx = 0
	examples = []
	total = 0
	answers = []
	with open(out_file, 'w') as of:
		for idx, data in enumerate(datas):
			eid = data['eid']

			passage = data['passage']
			candidates = data['candidates']
			answers = data['answer_sequence']
			num_blanks = data['number_of_blanks']
			num_candidates = data['candidate_length']
			blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
			
			for b_idx, a_idx in zip(blank_indexes, answers):
				if data_type == 'train':
					if b_idx - 1 in blank_indexes:
						total += 1
						continue
				for c_idx, candi in enumerate(candidates):
					if c_idx == a_idx[1]:
						label = '0'
					else:
						label = '1'
					next_id = b_idx + 1
					all_previous =  ' '.join(passage[:b_idx]).strip().split()
					quesiton_idx += 1





def single_previous(in_file, out_file, data_type):
	with open(in_file) as f:
		datas = json.load(f)
	#print(len(datas))
	quesiton_idx = 0
	examples = []
	total = 0
	answers = []
	with open(out_file, 'w') as of:
		for idx, data in enumerate(datas):
			eid = data['eid']
			passage = data['passage']
			candidates = data['candidates']
			answers = data['answer_sequence']
			num_blanks = data['number_of_blanks']
			num_candidates = data['candidate_length']
			blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
			for b_idx, a_idx in zip(blank_indexes, answers):
				for c_idx, candi in enumerate(candidates):
					if c_idx == a_idx[1]:
						label = '0'
					else:
						label = '1'
					of.write('\t'.join((str(quesiton_idx), passage[b_idx-1], candi, label)) + '\n')
					quesiton_idx += 1




if __name__ == "__main__" :


	parser = argparse.ArgumentParser(description='Extractor context features from raw json file.')
	parser.add_argument('--output_dir', required=True,
						help='output folder of extracted features')
	parser.add_argument('--input_dir', required=True,
						help='json file folder')
	parser.add_argument('--feature_type', required=True,
						help='feature type, sp, sn, ap, an, spn, apn.')
	features_types = ['sp', 'sn', 'ap', 'an', 'spn', 'apn']
	args = parser.parse_args()
	if args.feature_type not in features_types:
		print('feature type is not valid, only select in [sp, sn, ap, an, spn, apn]')
	fea_type = args.feature_type
	if not os.path.exists(args.output_dir):
	    os.makedirs(args.output_dir)
	for data_type in ['train', 'dev', 'test']:
		print('processing %s data' % data_type)
		in_file = '.'.join((data_type, 'json'))
		in_file = os.path.join(args.input_dir, in_file)
		out_file = '.'.join((data_type, 'tsv'))
		out_file = os.path.join(args.output_dir, out_file)
		if fea_type == 'apn':
			all_prev_next(in_file, out_file, data_type)
		elif fea_type == 'ap':
			all_previous(in_file, out_file, data_type)
		elif fea_type == 'an':
			all_next(in_file, out_file, data_type)
		elif fea_type == 'sp':
			single_previous(in_file, out_file, data_type)
		elif fea_type == 'sn':
			single_next(in_file, out_file, data_type)
		elif fea_type == 'spn':
			single_prev_next(in_file, out_file, data_type)
		print('done')


