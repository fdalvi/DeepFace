import matplotlib.pyplot as plt
import re

def main():
	x = range(1,43)
	y = [0.923251, 0.80175, 0.6545, 0.55475, 0.83825, 0.944751, 0.947501, 0.869, 0.890876, 0.976876, 0.53025, 0.77625, 0.576125, 0.85625, 0.832624, 0.647, 0.65475, 0.783499, 0.686999, 0.704375, 0.588125, 0.83125, 0.84575, 0.945251, 0.784874, 0.952126, 0.77725, 0.801249, 0.869875, 0.81475, 0.9, 0.746249, 0.583, 0.8365, 0.84775, 0.919501, 0.828625, 0.86375, 0.641625, 0.882001, 0.814125, 0.849375]
	fig, ax = plt.subplots()
	# MAX_SAMPLES = 1000
	plt.bar(x, y, width=0.9, align="center")
	ax.set_xticks(x)
	ax.set_xticklabels([int(x) for x in range(1,43)])
	plt.xlabel("Facial attributes")
	plt.ylabel("Classification accuracy")
	plt.show()

if __name__ == '__main__':
	main()