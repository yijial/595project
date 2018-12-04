import numpy as np
import torch
import torch.optim
import torch.nn as nn
import sys
import os 
from sklearn.utils import shuffle
from scipy import sparse

# directory = "toronto_npz/"


def main(dir):
	# dimensions of the nn
	n_in, n_h1, n_h2, n_out = 42288, 100, 100, 1
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

	# train the model on training data
	for filename in os.listdir(dir):
		if filename.startswith("train"):
			print("training on: " + filename)
			data = sparse.load_npz(dir + '/' + filename).toarray()
			# print(data.shape)
			shuffle(data)
			# business_id, useful, review_count, bag_of_words
			label = torch.from_numpy(data[:,1].reshape(-1,1)).float()
			features = torch.from_numpy(data[:, 2:]).float()
			# print(label.shape)
			# print(features.shape)
			for epoch in range(100):
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


	standard = np.empty((0,1))
	pred = np.empty((0,1))
	bid = np.empty((0,1))
	# predict labels for test data
	for filename in os.listdir(dir):
		if filename.startswith("test"):
			print("generating label on: " + filename)
			data = sparse.load_npz(dir + '/' + filename).toarray()
			label = torch.from_numpy(data[:,1].reshape(-1,1)).float()
			features = torch.from_numpy(data[:, 2:]).float()
			label_pred = model(features)
			# loss = criterion(label_pred, label)
			label_pred = label_pred.detach().numpy().reshape(-1,1)
			bid = np.concatenate((bid, data[:,0].reshape(-1,1)),axis=0)
			standard = np.concatenate((standard, label.numpy().reshape(-1,1)),axis=0)
			pred = np.concatenate((pred, label_pred), axis=0)

	# format: golden label, predicted, business_id
	output = np.concatenate((bid, standard, pred), axis=1)
	csr_matrix = sparse.csr_matrix(output)
	print(output.shape)
	sparse.save_npz("nn_output.npz", csr_matrix)


if __name__ == "__main__":
	main(sys.argv[1])

