import numpy as np
import torch
import torch.optim
import torch.nn as nn
import sys
from sklearn.utils import shuffle


def main(train, test):
	data = np.load(train)
	# print(data.shape)
	shuffle(data)
	# business_id, useful, review_count, bag_of_words
	label = torch.from_numpy(data[:,1].reshape(-1,1)).float()
	features = torch.from_numpy(data[:, 2:]).float()
	# print(label.shape)
	# print(features.shape)

	# dimensions of the nn
	n_in, n_h1, n_h2, n_out = features.shape[1], 100, 100, 1
	# condition of converge
	thres = 0.001
	model = nn.Sequential(nn.Linear(n_in, n_h1),
			nn.Tanh(),
			nn.Linear(n_h1, n_h2),
			nn.Sigmoid(),
			nn.Linear(n_h2, n_out),
			# nn.Sigmoid()
			)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	prev_loss = 0

	for epoch in range(10000):
		y_pred = model(features)

		loss = criterion(y_pred, label)
		if(epoch % 100 == 0):
			print('epoch: ', epoch,' loss: ', loss.item())
			# converges, break
		
		if abs(prev_loss - loss.item()) <= thres:
			break
		prev_loss = loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	testdata = np.load(test)
	test_label = torch.from_numpy(testdata[:, 1].reshape(-1,1)).float()
	test_feature = torch.from_numpy(testdata[:, 2:]).float()
	label_pred = model(test_feature)
	loss = criterion(label_pred, test_label)
	label_pred = label_pred.detach().numpy().reshape(-1,1)
	
	# format: golden label, predicted, business_id
	output = np.concatenate((testdata[:, 1].reshape(-1,1), label_pred), axis=1)
	output = np.concatenate((output, testdata[:, 0].reshape(-1,1)), axis=1)
	print(output.shape)
	print('test loss: ' + str(loss.item()))
	np.save("nn_output.npy", output)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])

