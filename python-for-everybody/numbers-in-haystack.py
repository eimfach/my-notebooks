import os
import re

handle = open(os.getcwd() + "/access-web-data/assignment-1/text.txt")

def sumNumbersInFile(handle):
    theSum = 0
    contents = handle.read()
    numbers = re.findall('[0-9]+', contents)
    for num in numbers:
        theSum = theSum + int(num)
    
    return theSum

def sumNumbersInFileBrave(handle):
    return sum( [ int(num) for num in re.findall('[0-9]+', handle.read()) ] )

theSum = sumNumbersInFileBrave(handle)

print(theSum)