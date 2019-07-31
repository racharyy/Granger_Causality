# Code

`exp_lambda_analysis.py` contains all the analysis with fitting the search time interval with expoential distributions. It also includes extracting the lambda feature vectors for each user.

`plot_time_interval.py` contains codes to analysis and visualize the raw search time interval.

`data_loader.py` contains the loader that parses frequency vectors for each user. There are two parsing methods: `parse()` and `parse_by_month()`, please check the code comments for details.

# Data


1. `low_freq_tensors_calendar.pkl` and `not_low_freq_tensors_calendar.pkl` are to lists of numpy matrix. Each numpy matrix represents the frequency vectors packed together, in the shape of `[27, time_step]`. Each category frequency is weighted by the confidence provided by the NLP toolkit.

2. `compound_vectors_self_esteem.pkl` contains both the categorical and lambda feature vectors (concatenated) for each user in the LS and NLS groups. Each list in this pickle file is a list of tuples: `List[tuple(user_id, 54-d lambda numpy array), tuple(user_id, 54-d lambda numpy array), ...]` where 54 comes from concatenation of two 27-d feature vectors (categorical first and lambda later). 

There are 49 samples in LS group and 43 in NLS group. Notice that some user may only search a particular category once in his or her whole data, thus time interval cannot be calculated. These slots in the lambda vectors are replaced with 0.

Please load it with
```
from io import open
import pickle
with open('compound_vectors_self_esteem.pkl', 'rb') as f:
	(compound_ls_list, compound_nls_list) = pickle.load(f)
```

Notice that the lambda features are scaled by 10^5 ALREADY! You can adjust the scaling in the `generate_compound_features_with_ID` in `exp_lambda_analysis.py`

3. `compound_vectors_psi.pkl` contains similar lists of tuples with user id reference and 54-d compound vectors.

There are 24 psi samples and 64 npsi samples. 

Please load it with
```
from io import open
import pickle
with open('compound_vectors_psi.pkl', 'rb') as f:
	(psi, npsi) = pickle.load(f)
```