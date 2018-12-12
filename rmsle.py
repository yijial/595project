import numpy as np
from scipy import sparse
import sys

def rmsle(y_pred, y_true):
	return np.sqrt(np.mean(np.power((np.log(y_pred + 1) - np.log(y_true + 1)), 2)))


def main(filename):
	data = np.load(filename)

	y_true = data[:, 0]
	y_pred = data[:, 1]

	print(len(y_pred))
	print("RMSLE is:", rmsle(y_pred, y_true))


if __name__ == '__main__':
    main(sys.argv[1])