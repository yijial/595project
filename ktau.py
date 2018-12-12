import os
import numpy as np
from scipy import stats
from scipy import sparse
import math
import sys

def main(filename):
	scores = []

	baseline_data = np.load(filename)
	business_dict = dict() 
	# business_dict[business_id] = [[useful_counts...], [scores...]]

	for row in baseline_data:
		# useful count, baseline score, business id
		useful_count = row[0] 
		baseline_score = row[1]
		business_id = row[2] 
		if business_id not in business_dict:
			business_dict[business_id] = []
			business_dict[business_id].append([])
			business_dict[business_id].append([])
		business_dict[business_id][0].append(useful_count)
		business_dict[business_id][1].append(baseline_score)

	for business_id in business_dict:
		# pred_ranking = stats.rankdata(business_dict[business_id][1], method='ordinal')
		# ranking = stats.rankdata(business_dict[business_id][0], method='ordinal')
		ranking = business_dict[business_id][0]
		pred_ranking = business_dict[business_id][1]
		if len(ranking) > 1:
			tau, _ = stats.kendalltau(ranking, pred_ranking)
			# print(ranking)
			# print(pred_ranking)
		else:
			tau = 1
		# business_id, tau
		if not (math.isnan(tau)):
			scores.append(tau)
	print("Average kendall-tau score: ", sum(scores) / len(scores))

if __name__ == "__main__":
	main(sys.argv[1])