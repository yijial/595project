import os
import numpy as np
from scipy import stats

directory = "review_npy/"
tau_file = open("kendall_tau_score.txt","w")
scores = []

for filename in os.listdir(directory):
	baseline_data = np.load(directory + filename)
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
		ranking = stats.rankdata(business_dict[business_id][0], method='ordinal')
		pred_ranking = stats.rankdata(business_dict[business_id][1], method='ordinal')
		if len(ranking) > 1:
			tau, _ = stats.kendalltau(ranking, pred_ranking)
		else:
			tau = 1
		# business_id, tau
		tau_file.write(str(business_id)+"\t"+str(tau)+"\n")
		scores.append(tau)
print("Average kendall-tau score: ", sum(scores) / len(scores))
tau_file.close()