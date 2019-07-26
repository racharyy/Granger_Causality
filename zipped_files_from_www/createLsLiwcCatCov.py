import spacy
import numpy as np

nlp = spacy.load('en_core_web_lg')

all_train = [u'Authentic',u'words per search',u'Six letter',u'they',u'auxilary verb',u'negate',u'adjective',u'number',u'affect',u'posemo',u'anxious',u'sad',u'family',u'friend',u'cognitive',u'insight',u'cause',u'differ',u'see',u'biological process',u'body',u'health',u'drives',u'affiliation',u'achieve',u'risk',u'focus future',u'relativity',u'motion',u'space',u'time',u'money',u'informal',u'swear',u'Chat Abbreviations',u'Nonfluencies',u'Punctuations',u'question mark',u'dash',u'apostrophe']

nlp_liwc_cats = [nlp(i) for i in all_train]


matrix = []
for i,cat1 in enumerate(nlp_liwc_cats):
	row = []
	for j,cat2 in enumerate(nlp_liwc_cats):
		score = abs(cat1.similarity(cat2))
		if j!=i and score == 0:
			score = 0.01
		row.append(score)

	matrix.append(row)


matrix = np.array(matrix)
print matrix.shape

np.savetxt('ls_liwc_cov.csv', matrix, delimiter=',')