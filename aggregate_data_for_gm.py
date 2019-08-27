import pickle
import glob

psi_file = "Boyu/compound_vectors_psi.pkl"
se_file = "Boyu/compound_vectors_self_esteem.pkl"

# comp_vec_psi = pickle.load( open( psi,"rb" ) )

# for t in comp_vec_psi:
# 	print(t)
# 	print(type(t))
# 	print(len(t))
# 	print('================')
# 	for i in t:
# 		print(i)
# 	break

si_lst = []
search_lst = []

# helper lst
si_uids = []
npsi_uids = []


with open(psi_file, 'rb') as f:
	(psi, npsi) = pickle.load(f)


	si_lst = []
	search_lst = []

	for (uid, lst) in psi:
		si_uids.append(uid)		
		
		si_lst.append(1)
		search_lst.append(lst[27:])

	for (uid, lst) in npsi:
		npsi_uids.append(uid)
		
		si_lst.append(0)
		search_lst.append(lst[27:])

# this is the order of the uids
si_npsi_uids = si_uids + npsi_uids

ls_uids = []
nls_uids = []
with open(se_file, 'rb') as f:
	(ls, nls) = pickle.load(f)
	for (uid, lst) in ls:
		ls_uids.append(uid)
	for (uid, lst) in nls:
		nls_uids.append(uid)

se_lst = []
for uid in si_npsi_uids:
	if uid in ls_uids:
		se_lst.append(1)
	if uid in nls_uids:
		se_lst.append(0)

print(se_lst)
print(si_lst)

data = {
	'search': search_lst,
	'si':si_lst,
	'se':se_lst
}

f = open("data_for_graphical_model.pkl","wb")
pickle.dump(data,f)
f.close()



# data = {

# 'search': [ [l1,l2,....,l27],[],.....]
# [l1,l2,....,l27] ==> from the 2nd set of 27 vectors
# 'si':[0,1,.....]

# 'se':[1,0,.....]



# }