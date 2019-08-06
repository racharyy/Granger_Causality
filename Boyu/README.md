# Code

1. `exp_lambda_analysis.py` contains all the analysis with fitting the search time interval with expoential distributions. It also includes extracting the lambda feature vectors for each user.
Scaling options are provided in the method `extract_lambda_feature_with_ID`, and we normally use hour (3600 times) as the scaling factor. Default setting is in minutes.

2. `plot_time_interval.py` contains codes to analysis and visualize the raw search time interval.

3. `data_loader.py` contains the loader that parses frequency vectors for each user. There are two parsing methods: `parse()` and `parse_by_month()`, please check the code comments for details.

4. `simple_mlp.py` contains a small two-layer MLP. It is able to find a meaningful hidden representation for the Lambda features in another dimension (like an automatic kernel projection). I was inspired by the pretraining process standard among NLP tasks. By setting hidden sizes to around 100 to 200, I obtained clusters showing a great pattern of separation vidsualized by tSNE. 

Notice that the MLP is only used as a pretraining method. In order to utilize the full capacity of it, we did not use as an end-to-end classifier. Instead, we train it by batch GD over the whole dataset and take the hidden layer as high dimensional representation.

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

4. `lambda_vectors_with_user_ID.pkl` is the data of the raw Lambdas extracted WITHOUT any scaling. The format is the same (list of tuples) as above, containg user-to-lambda mapping.

5. `lambda_vectors_cleaned_3600.pkl` and `lambda_vectors_3600.pkl` are lambda data scaled by hours (3600 times) since the default unit is in seconds. It also produced the best pretraining performance. 

All the other files are temporary files or for testing purpose. Other instructions will ne added soon. 