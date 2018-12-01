import json
import numpy as np

review_file = '../yelp_academic_dataset_review.json'
business_file = '../yelp_academic_dataset_business.json'
user_score = {}
user_total = {}
business_list = {}
businesses= {}
filteredData = {}

with open(business_file, 'r') as f:
	for line in f.readlines():
		business = json.loads(line)
		if business['categories'] is not None and "Restaurants" in business['categories']:
			businesses[business['business_id']] = {'review_count': business['review_count'], 'city': business['city'].replace(' ', '').lower()}

i = 0
with open(review_file, 'r') as f:
	for line in f.readlines():
		review = json.loads(line)
		user_id = review['user_id']
		business_id = review['business_id']

		useful = review['useful']
		
		if useful > 0 and business_id in businesses:
			if user_id not in user_total:
				user_total[user_id] = 0
			user_total[user_id]+=1
			if user_id not in user_score:
				user_score[user_id] = 0
			user_score[user_id]+=useful
			review_count = businesses[business_id]['review_count']
			city = businesses[business_id]['city']

			if city not in filteredData:
				filteredData[city] = []
			output = {'useful': useful, 'user_id': user_id, 'business_id': business_id}
			if business_id not in business_list:
				business_list[business_id] = i
				i+=1
			filteredData[city].append(output)

for city, data in filteredData.items():
	result = []
	for review in data:
		review_array = [review['useful'], user_score[review['user_id']]/user_total[review['user_id']], business_list[review['business_id']]]
		result.append(review_array)
	result = np.array(result)
	np.save('review_'+city+'.npy', result)
