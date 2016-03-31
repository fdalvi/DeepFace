import matplotlib.pyplot as plt
import re
import numpy as np

def main():
	x = []
	y = []
	for i in xrange(42):
		y.append([])
	for line in open('train_log.txt'):
		if "accuracy" in line:
			matches = re.search("accuracy-(\d+) = (\d+\.\d+)", line)
			if matches is None:
				continue
			# if matches.group(1) == '1':
				# print matches.group(1), matches.group(2)
			y[int(matches.group(1))-1].append(float(matches.group(2)))

	x = range(1, len(y[0])+1)
	# print len(x)
	# print len(y)
	# print len(y[0]), len(y[1]), len(y[2])
	# print y[0]
	# MAX_SAMPLES = 1000
	# plt.hold(True)
	average_y = []
	for i in xrange(len(y[0])):
		average_y.append(0)
		for j in xrange(42):
			average_y[i] += y[j][i]
		average_y[i] /= 42.0
	for i in xrange(42):
		plt.plot(x, y[i], '--')
		# plt.plot(x, y[2])
	plt.plot(x, average_y, 'k-', linewidth=5)
	# plt.plot(x, y[1])
	# plt.hold(False)
	plt.xlabel("Iterations")
	plt.ylabel("Accuracy")
	plt.show()

if __name__ == '__main__':
	main()