# Predicting and Ranking Yelp Reviews By Usefulness

## Data Collection
Downloaded from [Yelp Academic Dataset](https://www.yelp.com/dataset) 
*raw data is not included due to the size and difficulty of upload. Data after preprocessing is included

## Preprocessing
\* Preprocessing can be skipped. Due to dataset size, raw data is not included in the folder. Only preprocessed data is provided.
1. 	Read raw yelp_academic_dataset_review.json and yelp_academic_dataset_business.json, split reviews and businesses based on city and extract attributes(review text, business id, business total review count) and labels(review useful vote). Output files reviews_cityname.json in directory. 
	`python extract.py yelp_raw_json_dir filtered_city_dir`

2. 	Preprocess reviews in Las Vegas and Toronto with tokenization and stemmization. 
 - Run preprocess_bag_of_words.py for transforming reviews into bag-of-words vectors, and outputing files train_x.npz for training set and files test_x.npz for testing set in directory. 
 	`python preprocess_bag_of_words.py filtered_city_dir cityname bow_dir`
 - Run preprocess_sequence.py for LSTM to transform reviews into sequence tokens, with each word corresponds to a sequence words. Output x_train.npy, x_test.npy, y_train.npy, y_test.npy, word_index.npy, bid_test.npy in directory. 
	`python preprocess_sequence.py filtered_city_dir cityname seq_dir`

## ML model. 
1.	SVM. Output prediction_svm_cityname.npy
 	`python SVM.py bow_dir cityname`

2.	Neural Network. Output prediction_nn_cityname.npy
 	`python model_nn.py bow_dir cityname`

3.	LSTM. Output prediction_lstm_cityname.npy
 	`python lstm.py seq_dir cityname`

## Evaluation
1. 	Calculate Kendall's Tau Coefficient. 
	`python ktau.py filename`

2.	Calculate Root Mean Squared Logarithmic Error. 
	`python rmsle.py filename`