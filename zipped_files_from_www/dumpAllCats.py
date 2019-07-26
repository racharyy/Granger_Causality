import glob
import json
import matplotlib.pyplot as plt
import pickle

train_lsFiles = glob.glob("train-low-self-esteem/*.json")
train_nlsFiles = glob.glob("train-not-low-self-esteem/*.json")
validation_lsFiles = glob.glob("validation-low-self-esteem/*.json")
validation_nlsFiles = glob.glob("validation-not-low-self-esteem/*.json")

train_siFiles = glob.glob("train-suicide/*.json")
train_nsiFiles = glob.glob("train-non-suicide/*.json")
validation_siFiles = glob.glob("validation-suicide/*.json")
validation_nsiFiles = glob.glob("validation-non-suicide/*.json")

train_ls_categories = {}
for subject in train_lsFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in train_ls_categories:
					train_ls_categories[top_cat] = 1
				else:
					train_ls_categories[top_cat] += 1

pickle.dump(train_ls_categories, open( "train_ls_cat_dic.pickle", "wb" ))

train_nls_categories = {}
for subject in train_nlsFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in train_nls_categories:
					train_nls_categories[top_cat] = 1
				else:
					train_nls_categories[top_cat] += 1

pickle.dump(train_nls_categories, open( "train_nls_cat_dic.pickle", "wb" ))



train_si_categories = {}
for subject in train_siFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in train_si_categories:
					train_si_categories[top_cat] = 1
				else:
					train_si_categories[top_cat] += 1

pickle.dump(train_si_categories, open( "train_si_cat_dic.pickle", "wb" ))


train_nsi_categories = {}
for subject in train_nsiFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in train_nsi_categories:
					train_nsi_categories[top_cat] = 1
				else:
					train_nsi_categories[top_cat] += 1

pickle.dump(train_nsi_categories, open( "train_nsi_cat_dic.pickle", "wb" ))
######################################################################
# VALIDATIONS
######################################################################
val_ls_categories = {}
for subject in validation_lsFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in val_ls_categories:
					val_ls_categories[top_cat] = 1
				else:
					val_ls_categories[top_cat] += 1

pickle.dump(val_ls_categories, open( "val_ls_cat_dic.pickle", "wb" ))

val_nls_categories = {}
for subject in validation_nlsFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in val_nls_categories:
					val_nls_categories[top_cat] = 1
				else:
					val_nls_categories[top_cat] += 1

pickle.dump(val_nls_categories, open( "val_nls_cat_dic.pickle", "wb" ))



val_si_categories = {}
for subject in validation_siFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in val_si_categories:
					val_si_categories[top_cat] = 1
				else:
					val_si_categories[top_cat] += 1

pickle.dump(val_si_categories, open( "val_si_cat_dic.pickle", "wb" ))


val_nsi_categories = {}
for subject in validation_nsiFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				if top_cat not in val_nsi_categories:
					val_nsi_categories[top_cat] = 1
				else:
					val_nsi_categories[top_cat] += 1

pickle.dump(val_nsi_categories, open( "val_nsi_cat_dic.pickle", "wb" ))
##################################################################
# all Cat
##################################################################
all_cats = train_ls_categories.keys()+train_nls_categories.keys()+train_si_categories.keys()+train_nsi_categories.keys()+val_ls_categories.keys()+val_nls_categories.keys()+val_si_categories.keys()+val_nsi_categories.keys()
pickle.dump(all_cats, open( "all_categories.pickle", "wb" ))