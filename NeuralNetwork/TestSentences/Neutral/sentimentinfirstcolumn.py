# Set sentiment in the first column of all tweets
# Nathalie Stierman - S2544229
# python3 positivetweets.py name_of_old_file.txt

import sys

def main(argv):
	# Open files
	with open(argv[1], "r") as f:
		data = f.readlines()
		f.close()
	
	#count = 0
	file = open("newpositivetweets.txt", "w")
	for line in data:
		sentiment = line.split()
		# if the sentence is marked as positive, place tweet in file
		if sentiment[4] == 'neutral' or 'neutral' in sentiment[2]:
			#print(sentiment[0], "sentiment 0")
			#sentiment[0] = 'positive'
			#print(sentiment[0], "sentiment 0 hierna")
			newline = 'neutral'+ ' ' + line
			print(newline)
			file.write(newline)
			#count += 1
		
if __name__ == "__main__":
	if len(sys.argv) == 2:
		main(sys.argv)
