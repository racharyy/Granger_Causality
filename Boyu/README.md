# Code

`exp_lambda_analysis.py` contains all the analysis with fitting the search time interval with expoential distributions. It also includes extracting the lambda feature vectors for each user.

`plot_time_interval.py` contains codes to analysis and visualize the raw search time interval.

`data_loader.py` contains the loader that parses frequency vectors for each user. There are two parsing methods: `parse()` and `parse_by_month()`, please check the code comments for details.

# Data

1. 
`lambda_data_per_user.pkl` contains the lambda feature vectors for each user. Please load it in the form of a tuple `(low_matrix, not_low_matrix)` with python pickle package. These matrices are numpy matrices with shapes: `[low: (54, 27), not_low: (38, 27)]` (after eliminating invalid date).

2. 
`lambda_vectors_with_user_ID.pkl` has reference with user ID. 

Please load it with
```
from io import open
import pickle
with open('lambda_vectors_with_user_ID.pkl', 'rb') as f:
	(low_list, not_low_list) = pickle.load(f)

```

Each of the list is a list of tuples. Concretely, `List[tuple(user_id, 27-d lambda numpy array), tuple(user_id, 27-d lambda numpy array), ...]` where the langth is the number of people in LS and NLS, respectively.

Notice that some user may only search a particular category once in his or her whole data, thus time interval cannot be calculated. These slots are replaced with 0.

3. 
`low_freq_tensors_calendar.pkl` and `not_low_freq_tensors_calendar.pkl` are to lists of numpy matrix. Each numpy matrix represents the frequency vectors packed together, in the shape of `[27, time_step]`. Each category frequency is weighted by the confidence provided by the NLP toolkit.