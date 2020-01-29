import os

handle = open(os.getcwd() + "/mbox-short.txt")

########### Most prolific committer ########

def mostProlificCommitter():
	commitHistogram = dict()

	for line in handle:
		if line.find("From ") > - 1 :
			words = line.split()
			committer = words[1]
			commitHistogram[committer] = commitHistogram.get(committer, 0) + 1

	mostCommits = None
	mostProlificCommitter = None

	for currentCommitter,pushedCommits in commitHistogram.items():

		if mostProlificCommitter is None or pushedCommits > mostCommits:
			mostProlificCommitter = currentCommitter

		if mostCommits is None or pushedCommits > mostCommits:
			mostCommits = pushedCommits

	print(mostProlificCommitter, mostCommits)

def accumulateCommitHours():
	commitHoursHistogram = dict()

	for line in handle:
		
		if line.find("From ") > - 1:

			words = line.split()
			word = words[5]

			time = word.split(":")
			hourOfDay = time[0]

			if len(hourOfDay) == 1:
				hourOfDay = "0" + hourOfDay

			commitHoursHistogram[hourOfDay] = commitHoursHistogram.get(hourOfDay, 0) + 1

	sortedHistogram = sorted(commitHoursHistogram.items())
	for k,v in sortedHistogram:
		print(k,v)


handle.close()