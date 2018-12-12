import numpy as np
import torch
import torch.optim
import torch.nn as nn
import sys
import os 
from sklearn.utils import shuffle
from scipy import sparse

def train_model(model, criterion, optimizer, directory):
	# train the model on training data
	for filename in os.listdir(directory):
		if filename.startswith("train"):
			print("training on: " + filename)
			data = sparse.load_npz(directory + '/' + filename).toarray()
			# print(data.shape)
			data = shuffle(data)
			# business_id, useful, review_count, bag_of_words
			label = torch.from_numpy(data[:,1].reshape(-1,1)).float()
			# print(label.detach())
			features = data[:, 2:]
			# print(np.sum(features,axis=1))
			features = torch.from_numpy(features).float()
			# print(label.shape)
			# print(features.detach())
			for i in range(2):
				y_pred = model(features)

				loss = criterion(y_pred, label)
				if(i % 1 == 0):
					print('iteration: ', i,' loss: ', loss.item())
					# print(y_pred.detach()[1:10])
					# for param in model.parameters():
					# 	print(param.detach())
					# print(model.parameters().data())
					# converges, break
				
				# if abs(prev_loss - loss.item()) <= thres:
				# 	break
				# prev_loss = loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


def main(bow_dir, cityname):
	directory = bow_dir+"/"+cityname
	dim_data = sparse.load_npz(directory + '/' + "train.npz").toarray()
	# dimensions of the nn
	n_in, n_h1, n_h2, n_out = dim_data.shape[1] - 2, 1000, 1000, 1
	# condition of converge
	thres = 0.001
	model = nn.Sequential(nn.Linear(n_in, n_h1),
		# nn.Tanh(),
		# nn.Linear(n_h1, n_h2),
		nn.Sigmoid(),
		nn.Linear(n_h1, n_out),
		# nn.Sigmoid()
		)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	# prev_loss = 0
	for epoch in range(10):
		train_model(model, criterion, optimizer, directory)

	standard = np.empty((0,1))
	pred = np.empty((0,1))
	bid = np.empty((0,1))
	# predict labels for test data
	for filename in os.listdir(directory):
		if filename.startswith("test"):
			print("generating label on: " + filename)
			data = sparse.load_npz(directory + '/' + filename).toarray()
			label = torch.from_numpy(data[:,1].reshape(-1,1)).float()
			features = torch.from_numpy(data[:, 2:]).float()
			label_pred = model(features)
			# loss = criterion(label_pred, label)
			label_pred = label_pred.detach().numpy().reshape(-1,1)
			bid = np.concatenate((bid, data[:,0].reshape(-1,1)),axis=0)
			standard = np.concatenate((standard, label.numpy().reshape(-1,1)),axis=0)
			pred = np.concatenate((pred, label_pred), axis=0)

	# format: golden label, predicted, business_id
	output = np.concatenate((standard, pred, bid), axis=1)
	print(output.shape)
	np.save("prediction_nn_" + cityname + ".npy", output)


if __name__ == "__main__":
	main(sys.argv[1],sys.argv[2])

