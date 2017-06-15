# Place all positive tweets inside a new file
# Nathalie Stierman - S2544229
# python3 positivetweets.py name_of_old_file.txt

import sys

def main(argv):
	# Open files
	with open(argv[1], "r") as f:
		data = f.readlines()
		f.close()
	
	count = 0
	for line in data:
		sentiment = line.split()
		# if the sentence is marked as positive, place tweet in new file
		if sentiment[4] == 'positive' or 'positive' in sentiment[2]:
			file = open("positive_{}".format(count), "w")
			print(line)
			file.write(line)
			count += 1
		
if __name__ == "__main__":
	if len(sys.argv) == 2:
		main(sys.argv)
