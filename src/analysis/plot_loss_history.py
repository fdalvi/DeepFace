import matplotlib.pyplot as plt
import re

def main():
	x = []
	y = []
	for line in open('train_log_conv5_3.txt'):
		if "loss = " in line:
			print line
			x.append(int(re.search("Iteration (\d+),", line).group(1)))
			y.append(float(re.search("loss = (\d+(\.\d+)?)", line).group(1)))

	MAX_SAMPLES = 1000
	plt.plot(x[:MAX_SAMPLES], y[:MAX_SAMPLES])
	plt.xlabel("Num iterations")
	plt.ylabel("loss")
	plt.show()

if __name__ == '__main__':
	main()