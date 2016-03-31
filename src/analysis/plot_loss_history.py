import matplotlib.pyplot as plt
import re

def main():
	colors = ['-b', '-r', '-g']
	labels = ['fc6', 'conv5', 'conv3']
	plots = []
	for t, trial in enumerate(['train_log.txt', 'train_log_conv5_3.txt', 'train_log_conv3_1.txt']):
		x = []
		y = []
		for line in open(trial):
			if "loss = " in line:
				x.append(int(re.search("Iteration (\d+),", line).group(1)))
				y.append(float(re.search("loss = (\d+(\.\d+)?)", line).group(1)))
		print len(y)
		MAX_SAMPLES = 1000
		plots.append(plt.plot(x[:MAX_SAMPLES], y[:MAX_SAMPLES], colors[t], label=labels[t]))
	plt.xlabel("Num iterations")
	plt.ylabel("loss")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()