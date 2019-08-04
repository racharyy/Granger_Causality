from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# link to vader
# https://github.com/cjhutto/vaderSentiment

def sentiment_scores(sentence): 
	'''
	Given a sentence, this function returns a dictionary
	with the positive, negative, neutral and compound 
	sentiment. For example it returns a dic like this:

	{
		'pos': 0.487, 
		'compound': 0.431, 
		'neu': 0.513, 
		'neg': 0.0
	}
	'''
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 

    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict