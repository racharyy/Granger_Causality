import glob
import json

lsFiles = glob.glob("train-low-self-esteem/*.json")
nlsFiles = glob.glob("train-not-low-self-esteem/*.json")
all_categories = {}

for subject in lsFiles:
	with open(subject, 'r') as f:
		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]
				print top_cat
	break