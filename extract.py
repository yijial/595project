import json

review_file = 'yelp_academic_dataset_review.json'
business_file = 'yelp_academic_dataset_business.json'
filteredData = {}
data = []
businesses = {}

with open(business_file, 'r') as f:
	for line in f.readlines():
		business = json.loads(line)
		if business['categories'] is not None and "Restaurants" in business['categories']:
			businesses[business['business_id']] = {'review_count': business['review_count'], 'city': business['city'].replace(' ', '').lower()}

with open(review_file, 'r') as f:
	for line in f.readlines():
		review = json.loads(line)
		useful = review['useful']
		text = review['text']
		business_id = review['business_id']
		if useful > 0 and business_id in businesses:
			city = businesses[business_id]['city']
			review_count = businesses[business_id]['review_count']

			if city not in filteredData:
				filteredData[city] = []
			output = {'text': text, 'useful': useful, 'review_count': review_count, 'business_id': business_id}
			filteredData[city].append(output)

for city, data in filteredData.items():
	with open('review_'+city+'.json', 'w') as f:
		json.dump(data, f)
