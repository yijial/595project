import numpy as np
import torch
import torch.optim
import torch.nn as nn
import sys

def main(file):
	data = np.load(file)
	# print(data.shape)
	label = torch.from_numpy(data[:,0].reshape(-1,1)).float()
	features = torch.from_numpy(data[:,1:]).float()
	# print(label.shape)
	# print(features.shape)

	# dimensions of the nn
	n_in, n_h, n_out = features.shape[1], 100, 1
	# condition of converge
	thres = 0.0001
	model = nn.Sequential(nn.Linear(n_in, n_h),
			nn.ReLU(),
			nn.Linear(n_h, n_out),
			nn.Sigmoid())

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	prev_loss = 0

	for epoch in range(1000):
		y_pred = model(features)

		loss = criterion(y_pred, label)
		if(epoch % 100 == 0):
			print('epoch: ', epoch,' loss: ', loss.item())

		if abs(prev_loss - loss.item()) <= thres:
			break

		prev_loss = loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


if __name__ == "__main__":
	main(sys.argv[1])

