# Author: Sameer Darekar
# Title: Implementing Apriori Algorithm
# Dataset Location: https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data

"""
   Attribute Information
1. Wife's age (numerical)
2. Wife's education (categorical) 1=low, 2, 3, 4=high
3. Husband's education (categorical) 1=low, 2, 3, 4=high
4. Number of children ever born (numerical)
5. Wife's religion (binary) 0=Non-Islam, 1=Islam
6. Wife's now working? (binary) 0=Yes, 1=No
7. Husband's occupation (categorical) 1, 2, 3, 4
8. Standard-of-living index (categorical) 1=low, 2, 3, 4=high
9. Media exposure (binary) 0=Good, 1=Not good
10. Contraceptive method used (class attribute) 1=No-use, 2=Long-term, 3=Short-term
"""

import numpy as np
import itertools

def readDataset():
	path=raw_input("Enter the Path:")
	titles=['1','2','3','4','5','6','7','8','9','10']
	lexicon=set()
	dictData={}
	f=open(path)
	line="text"
	i=1
	while line:
		line=f.readline().strip()
		vector=line.split(',')
		if len(vector)==1:
			continue
		# converting continous to categorical
		temp=int(vector[0])
		if temp>=10 and temp<=20:
			vector[0]='1'
		elif temp>20 and temp<=30:
			vector[0]='2'
		elif temp>30 and temp<=40:
			vector[0]='3'
		elif temp>40 and temp<=50:
			vector[0]='4'
		#
		temp=int(vector[3])
		if temp>=0 and temp<=4:
			vector[3]='1'
		elif temp>4 and temp<=8:
			vector[3]='2'
		elif temp>8 and temp<=12:
			vector[3]='3'
		elif temp>12 and temp<=16:
			vector[3]='4'

		newVector=[titles[j]+'_'+vector[j] for j in range(len(titles))]
		lexicon.update(newVector)
		dictData[i]=newVector
		i+=1
	f.close()
	lexicon=sorted(lexicon)
	return dictData,titles,lexicon

def tf(word,vector):
	return vector.count(word)

def buildVectors(data,lexicon):
	vectors={}
	for i in range(1,len(data)+1):
		vectors[i]=[tf(word,data[i]) for word in lexicon]
	return vectors

def getf1(vectors,minsupport):
	f1Itemset={}
	for i in range(0,len(vectors[0])):
		temp=vectors[:,i].tolist()
		head=temp[0]
		temp=temp[1:]
		sumAll=temp.count('1')
		f1Itemset[head]=sumAll
		i+=1
	f1Itemset={key: value for key,value in f1Itemset.items() if value>0} #remove itemset if their value is 0
	print "#"*70
	print "for F1 itemset"
	print "#"*70
	print "number of generated candidate itemsets: ",len(f1Itemset)
	f1Itemset={key: value for key,value in f1Itemset.items() if value>minsupport}
	print "number of Frequent itemsets: ",len(f1Itemset)
	print "#"*70
	return f1Itemset

# def convertVectors(vectors):
# 	convertedVec=[]
# 	for vector in vectors:
# 		convertedVec.append([vector.index(x) for x in vector])
# 	return convertedVec

def getCount(itemset,vectors):
	f2itemset=[]
	lexicon=vectors[0].tolist()
	vectors=vectors[1:]
	for candidate in itemset:
		candCount=0
		indexVec=[lexicon.index(x) for x in candidate]
		for vector in vectors:
			# if (set(indexVec)<set(vector)):
			# 	candCount+=1
			# 	f2itemset.append([candidate,candCount])
			count=0
			for indexi in indexVec:
				if vector[indexi]=='0':
					break
				else:
					count+=1
			if(count==len(candidate)):
				candCount+=1
		if candCount!=0:
			f2itemset.append([candidate,candCount])
	return f2itemset

def getDictItemset(f2itemset):
	dictItemset={}
	for candidate in f2itemset:
		dictkey=','.join(candidate[0])
		dictItemset[dictkey]=candidate[1]
	return dictItemset

def getf2(f1Itemset,vectors,minsupport):
	f1list=sorted(list(f1Itemset.keys()))
	f2Itemset=[list(x) for x in itertools.combinations(f1list,2)]
	f2Itemset=getCount(f2Itemset,vectors)
	f2Itemset=sorted(f2Itemset)
	f2dict=getDictItemset(f2Itemset) #convert to dictionary for simplicity of operations
	print "#"*70
	print "for F2 itemset"
	print "#"*70
	print "number of generated candidate itemsets: ",len(f2dict)
	f2dict={key: value for key,value in f2dict.items() if value>minsupport}
	maximalFreqItemset=getmaxFreqItemsets(f2dict,f1list)
	freqClosedItemset=getFreqClosedItemsets(f2dict,f1Itemset)
	print "number of Frequent itemsets: ",len(f2dict)
	print "*"*70
	print "Maximal Frequent itemset count : ",len(maximalFreqItemset)
	print "maximal Frequent itemsets are : ",maximalFreqItemset
	print "*"*70
	print "Closed Frequent itemset count : ",len(freqClosedItemset)
	print "Closed Frequent itemsets are : ",freqClosedItemset
	print "#"*70
	return f2dict


def checkSuperSet(k,k_1):
	if type(k_1)!=list:
		k_1=[k_1]
	count=0
	for element in k_1:

		if element in k:
			count+=1
		else:
			break
	if count==len(k_1):
		return True
	else:
		return False

def getmaxFreqItemsets(reqItemsetDict,itemsetk_1):
	maximalFreqItemset=[]
	setk_1=[]
	reqItemset=[]
	for candidate in reqItemsetDict:
		reqItemset.append(candidate.split(','))
	reqItemset=sorted(reqItemset)
	itemsetk_1=sorted(itemsetk_1)
	for candidatek_1 in itemsetk_1:
		if type(itemsetk_1[0])==list: #for non F1 itemset
			setk_1.append(','.join(candidatek_1))
		else:
			setk_1.append(candidatek_1) # for F1 itemset
		count=0
		for candidatek in reqItemset:
			#if set(candidatek).issuperset(set(candidatek_1)):
			if (checkSuperSet(candidatek,candidatek_1)):
				break
			count+=1
		if count==len(reqItemset):
			maximalFreqItemset.append(candidatek_1)
	# maximalFreqItemset=set(setk_1)-set(nonMaxFreqItemset)
	# maximalFreqItemset=list(maximalFreqItemset)
	return maximalFreqItemset

def getFreqClosedItemsets(itemsetk,itemsetk_1):
	listk=[]
	listk_1=[]
	setk_1=[]
	closedItemset=[]
	for candidate in itemsetk:
		listk.append(candidate.split(','))
	for candidate in itemsetk_1:
		listk_1.append(candidate.split(','))
	for k_1 in listk_1:
		setk_1.append(','.join(k_1))
		supportk_1=itemsetk_1[','.join(k_1)]
		for k in listk:
			supportk=itemsetk[','.join(k)]
			if checkSuperSet(k,k_1):
				if(supportk==supportk_1):
					closedItemset.append(','.join(k_1))
					break
	#closedFreqItemset=set(setk_1)-set(nonClosedItemset)
	return list(closedItemset)


def getf_k(k,vectors,itemsetList,minsupport):
	_1=itemsetList[1]
	k_1=itemsetList[(k-1)]
	itemsetk_1=[]
	itemset_1=[]
	reqItemset=[]
	for candidate in k_1:
		itemsetk_1.append(candidate.split(','))
	for candidate in _1:
		itemset_1.append(candidate)
	itemsetk_1=sorted(itemsetk_1)
	itemset_1=sorted(itemset_1)
	for itemk_1 in itemsetk_1:
		for item_1 in itemset_1:
			if itemk_1[k-2]<item_1:
				tempList=itemk_1+[item_1]
				reqItemset.append(tempList)
	reqItemset=getCount(reqItemset,vectors)
	reqItemsetDict=getDictItemset(reqItemset)
	if (len(reqItemsetDict)!=0):
		print "#"*70
		print "for F",k," itemset"
		print "#"*70
		print "number of generated candidate itemsets: ",len(reqItemsetDict)

		reqItemsetDict={key: value for key,value in reqItemsetDict.items() if value>minsupport}
		maximalFreqItemset=getmaxFreqItemsets(reqItemsetDict,itemsetk_1)
		freqClosedItemset=getFreqClosedItemsets(reqItemsetDict,itemsetList[k-1])
		print "number of Frequent itemsets: ",len(reqItemsetDict)
		print "*"*70
		print "Maximal Frequent itemset count : ",len(maximalFreqItemset)
		print "maximal Frequent itemsets are : ",maximalFreqItemset
		print "*"*70
		print "Closed Frequent itemset count : ",len(freqClosedItemset)
		print "Closed Frequent itemsets are : ",freqClosedItemset
		print "#"*70
	return reqItemsetDict

def fk_1(vectors,minsupport):
	itemsetList=[0]
	print"*********************"
	print"Using method Fk-1*F1"
	print"*********************"
	f1Itemset=getf1(vectors,minsupport) #build Frequent 1 itemset
	itemsetList.append(f1Itemset)
	if len(f1Itemset)>1:
		f2Itemset=getf2(f1Itemset,vectors,minsupport) #build Frequent 2 itemset
		itemsetList.append(f2Itemset)
	if len(f2Itemset)==0:
		return itemsetList
	k=3
	while True:
		itemsetList.append(getf_k(k,vectors,itemsetList,minsupport))
		k+=1
		if len(itemsetList[k-1])==0:
			break
	return itemsetList

def getf_k_1(k,vectors,itemsetList,minsupport):
	k_1=itemsetList[(k-1)]
	itemsetk_1=[]
	reqItemset=[]
	for candidate in k_1:
		itemsetk_1.append(candidate.split(','))
	itemsetk_1=sorted(itemsetk_1)
	for item1 in itemsetk_1:
		for item2 in itemsetk_1:
			if(cmp(item1[:k-3],item2[:k-3])==0):#if first k-2 items are equal
				if(item1[k-2]!=item2[k-2]):
					appendList=sorted(list(set(item1) | set(item2)))
					if len(appendList)==k:
						reqItemset.append(appendList)
	reqItemset=getCount(reqItemset,vectors)
	reqItemsetDict=getDictItemset(reqItemset)
	if(len(reqItemsetDict)!=0):
		print "#"*70
		print "for F",k," itemset"
		print "#"*70
		print "number of generated candidate itemsets: ",len(reqItemsetDict)
		reqItemsetDict={key: value for key,value in reqItemsetDict.items() if value>minsupport}
		maximalFreqItemset=getmaxFreqItemsets(reqItemsetDict,itemsetk_1)
		freqClosedItemset=getFreqClosedItemsets(reqItemsetDict,itemsetList[k-1])
		print "number of Frequent itemsets: ",len(reqItemsetDict)
		print "*"*70
		print "Maximal Frequent itemset count : ",len(maximalFreqItemset)
		print "maximal Frequent itemsets are : ",maximalFreqItemset
		print "*"*70
		print "Closed Frequent itemset count : ",len(freqClosedItemset)
		print "Closed Frequent itemsets are : ",freqClosedItemset
		print "#"*70
	return reqItemsetDict

def fk_1k_1(vectors,minsupport):
	itemsetList=[0]
	print"*********************"
	print"Using method Fk-1*Fk-1"
	print"*********************"
	f1Itemset=getf1(vectors,minsupport) #build Frequent 1 itemset
	itemsetList.append(f1Itemset)
	if len(f1Itemset)>1:
		f2Itemset=getf2(f1Itemset,vectors,minsupport) #build Frequent 2 itemset
		itemsetList.append(f2Itemset)
	if len(f2Itemset)==0:
		return itemsetList
	k=3
	while True:
		itemsetList.append(getf_k_1(k,vectors,itemsetList,minsupport))
		k+=1
		if len(itemsetList[k-1])==0:
			break
	return itemsetList

def main():
	data,titles,lexicon=readDataset()
	vectors=buildVectors(data,lexicon)
	vectors[0]=lexicon
	vectList=[]
	#convert dict to 2d list
	for key,value in vectors.iteritems():
		temp=value
		vectList.append(temp)
	vectList=np.array(vectList)
	minsupport=raw_input("Enter Minimum support %: ")
	minsupport=(len(vectList)-1)*(float(minsupport)/100)
	minsupport=int(minsupport)
	print "minimum support count : ",minsupport
	print"columns decoded as ColumnIndex_value"
	print"column names are as below"
	print lexicon
	itemsetList=fk_1(vectList,minsupport)
	# print itemsetList[3]
	#itemsetList=fk_1k_1(vectList,minsupport)


if __name__=="__main__":
	main()




#Output:
"""
C:\Users\samee\OneDrive\Assignments\DM assignent 4>python apriori_c_contraceptive.py
Enter the Path:contraceptive.csv
Enter Minimum support %: 10
minimum support count :  147
columns decoded as ColumnIndex_value
column names are as below
['10_1', '10_2', '10_3', '1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '5_0', '5_1', '6_0', '6_1', '7_1', '7_2', '7_3', '7_4', '8_1', '8_2', '8_3', '8_4', '9_0', '9_1']
*********************
Using method Fk-1*F1
*********************
######################################################################
for F1 itemset
######################################################################
number of generated candidate itemsets:  33
number of Frequent itemsets:  26
######################################################################
######################################################################
for F2 itemset
######################################################################
number of generated candidate itemsets:  301
number of Frequent itemsets:  151
**********************************************************************
Maximal Frequent itemset count :  26
maximal Frequent itemsets are :  ['6_0', '6_1', '8_4', '1_4', '1_3', '1_2', '10_1', '7_1', '7_3', '7_2', '3_3', '3_2', '3_4', '2_1', '2_2', '2_3', '2_4', '9_0', '5_1', '5_0', '4_2', '10_2', '4_1', '10_3', '8_2', '8_3']
**********************************************************************
Closed Frequent itemset count :  0
Closed Frequent itemsets are :  []
######################################################################
number of Frequent itemsets:  151
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  955
number of Frequent itemsets:  302
**********************************************************************
Maximal Frequent itemset count :  6
maximal Frequent itemsets are :  ['2_3,3_3', '1_3,4_2', '3_4,4_2', '3_2,5_1', '2_1,5_1', '5_0,6_1']
**********************************************************************
Closed Frequent itemset count :  1
Closed Frequent itemsets are :  ['10_2,2_4']
######################################################################
######################################################################
for F 4  itemset
######################################################################
number of generated candidate itemsets:  1015
number of Frequent itemsets:  259
**********************************************************************
Maximal Frequent itemset count :  37
maximal Frequent itemsets are :  ['2_3,6_1,7_3', '1_2,7_2,9_0', '5_1,7_3,8_4', '3_4,5_1,7_2', '2_3,4_1,7_3', '1_2,4_1,7_2', '10_2,7_1,9_0', '10_1,1_2,3_4', '3_3,6_1,7_3', '10_1,1_4,5_1', '10_1,7_2,9_0', '10_1,1_3,9_0', '7_2,8_4,9_0', '1_2,3_3,5_1', '1_2,2_2,5_1', '10_3,2_3,9_0', '4_1,5_1,8_2', '10_1,7_1,9_0', '10_1,5_1,7_2', '10_1,3_3,5_1', '1_2,3_3,9_0', '10_3,8_3,9_0', '1_4,3_4,9_0', '10_3,1_3,9_0', '1_4,8_4,9_0', '10_1,6_0,9_0', '3_4,6_0,8_4', '1_2,2_4,6_1', '1_3,7_3,9_0', '6_1,7_3,8_3', '5_1,8_2,9_0', '4_1,7_3,8_3', '5_0,8_4,9_0', '10_2,6_1,8_4', '7_3,8_4,9_0', '2_3,8_4,9_0', '5_1,6_1,8_2']
**********************************************************************
Closed Frequent itemset count :  7
Closed Frequent itemsets are :  ['10_2,2_4,5_1', '1_3,2_4,8_4', '10_2,2_4,3_4', '10_2,2_4,4_1', '1_3,2_4,5_1', '10_2,2_4,8_4', '3_4,6_1,8_3']
######################################################################
######################################################################
for F 5  itemset
######################################################################
number of generated candidate itemsets:  442
number of Frequent itemsets:  113
**********************************************************************
Maximal Frequent itemset count :  45
maximal Frequent itemsets are :  ['10_1,6_1,8_4,9_0', '3_4,4_1,7_2,9_0', '3_4,6_1,7_2,9_0', '10_3,4_1,6_1,7_3', '10_3,6_1,8_4,9_0', '3_4,5_1,6_0,9_0', '1_3,5_1,8_4,9_0', '4_1,5_1,6_0,9_0', '10_2,2_4,8_4,9_0', '1_2,2_3,6_1,9_0', '10_3,5_1,8_4,9_0', '1_3,2_4,5_1,9_0', '10_3,4_1,8_4,9_0', '1_2,6_1,8_3,9_0', '2_4,4_1,6_0,9_0', '10_3,3_4,8_4,9_0', '10_2,3_4,8_4,9_0', '10_2,4_1,5_1,9_0', '1_2,6_1,8_4,9_0', '10_1,2_2,5_1,9_0', '4_1,6_0,8_4,9_0', '3_4,4_1,5_1,6_0', '1_3,2_4,6_1,9_0', '2_2,5_1,7_3,9_0', '10_1,2_3,5_1,9_0', '1_3,3_4,7_1,9_0', '10_1,3_4,8_4,9_0', '10_2,5_1,6_1,9_0', '10_2,2_4,5_1,9_0', '2_4,3_4,6_0,9_0', '2_3,5_1,7_3,9_0', '1_4,5_1,6_1,9_0', '1_3,4_1,5_1,6_1', '4_2,5_1,6_1,9_0', '3_3,5_1,7_3,9_0', '5_1,7_3,8_3,9_0', '3_4,4_1,5_0,9_0', '2_3,3_4,4_1,9_0', '10_3,2_4,5_1,9_0', '10_2,3_4,5_1,9_0', '10_2,3_4,6_1,9_0', '10_2,4_1,6_1,9_0', '1_3,6_1,8_4,9_0', '10_1,5_1,8_3,9_0', '3_4,4_1,6_0,9_0']
**********************************************************************
Closed Frequent itemset count :  6
Closed Frequent itemsets are :  ['10_2,2_4,3_4,4_1', '3_4,4_1,6_1,8_3', '3_4,5_1,6_1,8_3', '1_3,2_4,3_4,8_4', '2_4,4_1,7_1,8_4', '2_4,6_1,7_1,8_4']
######################################################################
######################################################################
for F 6  itemset
######################################################################
number of generated candidate itemsets:  79
number of Frequent itemsets:  23
**********************************************************************
Maximal Frequent itemset count :  34
maximal Frequent itemsets are :  ['1_2,4_1,5_1,8_3,9_0', '1_3,3_4,4_1,5_1,9_0', '10_2,2_4,3_4,4_1,9_0', '10_1,2_4,3_4,4_1,9_0', '3_4,5_1,6_1,8_3,9_0', '3_4,5_1,6_1,7_3,9_0', '3_4,4_1,6_1,8_3,9_0', '10_3,5_1,6_1,7_3,9_0', '1_2,2_3,4_1,5_1,9_0', '10_1,5_1,6_1,7_3,9_0', '4_1,5_1,6_1,8_3,9_0', '3_4,4_1,5_1,6_1,7_3', '10_1,4_1,5_1,8_4,9_0', '10_3,2_4,3_4,4_1,9_0', '1_3,3_4,4_1,8_4,9_0', '1_2,4_1,5_1,8_4,9_0', '2_3,3_4,5_1,6_1,9_0', '1_3,3_4,4_1,6_1,9_0', '3_4,4_1,6_1,7_3,9_0', '3_4,4_1,5_1,7_3,9_0', '4_1,5_1,6_1,7_2,9_0', '1_3,3_4,5_1,6_1,9_0', '10_1,1_2,5_1,6_1,9_0', '1_3,2_4,3_4,4_1,9_0', '10_1,4_1,5_1,7_3,9_0', '10_3,4_1,5_1,7_3,9_0', '10_1,1_2,4_1,6_1,9_0', '2_3,4_1,5_1,6_1,9_0', '3_4,4_1,5_1,8_3,9_0', '1_3,2_4,3_4,8_4,9_0', '2_2,4_1,5_1,6_1,9_0', '10_1,1_2,4_1,5_1,9_0', '1_2,3_4,4_1,8_4,9_0', '3_3,4_1,5_1,6_1,9_0']
**********************************************************************
Closed Frequent itemset count :  2
Closed Frequent itemsets are :  ['2_4,4_1,7_1,8_4,9_0', '2_4,3_4,6_1,7_1,8_4']
######################################################################
######################################################################
for F 7  itemset
######################################################################
number of generated candidate itemsets:  1
number of Frequent itemsets:  0
**********************************************************************
Maximal Frequent itemset count :  23
maximal Frequent itemsets are :  ['2_4,3_4,4_1,5_1,7_1,9_0', '2_4,3_4,4_1,5_1,6_1,8_4', '10_3,1_2,4_1,5_1,6_1,9_0', '3_4,4_1,5_1,6_1,8_4,9_0', '2_4,3_4,5_1,7_1,8_4,9_0', '1_2,3_4,4_1,5_1,6_1,9_0', '3_4,4_1,5_1,6_1,7_1,9_0', '10_3,3_4,4_1,5_1,6_1,9_0', '2_4,3_4,4_1,6_1,8_4,9_0', '1_2,2_4,3_4,4_1,5_1,9_0', '3_4,5_1,6_1,7_1,8_4,9_0', '2_4,3_4,4_1,5_1,8_4,9_0', '2_4,3_4,5_1,6_1,7_1,9_0', '3_4,4_1,5_1,7_1,8_4,9_0', '2_4,3_4,4_1,5_1,6_1,9_0', '10_1,3_4,4_1,5_1,6_1,9_0', '3_4,4_1,6_1,7_1,8_4,9_0', '2_4,4_1,5_1,6_1,8_4,9_0', '2_4,3_4,5_1,6_1,8_4,9_0', '2_4,3_4,4_1,6_1,7_1,9_0', '2_4,3_4,4_1,7_1,8_4,9_0', '2_4,3_4,6_1,7_1,8_4,9_0', '1_2,4_1,5_1,6_1,7_3,9_0']
**********************************************************************
Closed Frequent itemset count :  0
Closed Frequent itemsets are :  []
######################################################################

C:\Users\samee\OneDrive\Assignments\DM assignent 4\Question3>PYTHON apriori_c_contraceptive.py
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\contraceptive.csv
Enter Minimum support %: 7
minimum support count :  103
columns decoded as ColumnIndex_value
column names are as below
['10_1', '10_2', '10_3', '1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '5_0', '5_1', '6_0', '6_1', '7_1', '7_2', '7_3', '7_4', '8_1', '8_2', '8_3', '8_4', '9_0', '9_1']
*********************
Using method Fk-1*F1
*********************
######################################################################
for F1 itemset
######################################################################
number of generated candidate itemsets:  33
number of Frequent itemsets:  28
######################################################################
######################################################################
for F2 itemset
######################################################################
number of generated candidate itemsets:  350
number of Frequent itemsets:  201
**********************************************************************
Maximal Frequent itemset count :  1
maximal Frequent itemsets are :  ['9_1']
**********************************************************************
Closed Frequent itemset count :  0
Closed Frequent itemsets are :  []
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  1579
number of Frequent itemsets:  499
**********************************************************************
Maximal Frequent itemset count :  9
maximal Frequent itemsets are :  [['10_3', '4_2'], ['1_3', '2_2'], ['1_3', '3_3'], ['1_3', '5_0'], ['2_2', '3_4'], ['2_2', '7_2'], ['2_2', '8_4'], ['3_2', '4_1'], ['4_2', '7_2']]
**********************************************************************
Closed Frequent itemset count :  1
Closed Frequent itemsets are :  ['10_2,2_4']
######################################################################
######################################################################
for F 4  itemset
######################################################################
number of generated candidate itemsets:  2288
number of Frequent itemsets:  567
**********************************************************************
Maximal Frequent itemset count :  42
maximal Frequent itemsets are :  [['10_1', '1_2', '7_3'], ['10_1', '1_3', '6_1'], ['10_1', '3_4', '7_3'], ['10_1', '4_2', '5_1'], ['10_1', '5_1', '8_2'], ['10_3', '6_0', '9_0'], ['10_3', '6_1', '7_2'], ['1_3', '2_3', '9_0'], ['1_3', '4_1', '6_0'], ['1_3', '4_1', '7_3'], ['1_3', '5_1', '6_0'], ['1_3', '5_1', '7_2'], ['1_3', '6_0', '9_0'], ['1_3', '7_2', '9_0'], ['1_4', '2_4', '8_4'], ['1_4', '3_4', '7_1'], ['1_4', '4_1', '5_1'], ['1_4', '4_1', '9_0'], ['1_4', '4_2', '5_1'], ['1_4', '4_2', '9_0'], ['1_4', '5_1', '7_1'], ['1_4', '7_1', '9_0'], ['2_1', '5_1', '6_1'], ['2_3', '4_1', '7_2'], ['3_2', '5_1', '6_1'], ['3_2', '5_1', '9_0'], ['3_2', '6_1', '9_0'], ['3_3', '5_1', '7_2'], ['3_3', '5_1', '8_3'], ['3_3', '7_2', '9_0'], ['3_3', '8_4', '9_0'], ['3_4', '6_0', '7_1'], ['4_2', '5_1', '7_3'], ['4_2', '7_3', '9_0'], ['5_1', '6_0', '7_3'], ['5_1', '6_1', '8_1'], ['5_1', '7_2', '8_3'], ['6_0', '7_1', '9_0'], ['6_0', '7_2', '9_0'], ['6_0', '7_3', '9_0'], ['7_1', '8_3', '9_0'], ['7_2', '8_3', '9_0']]
**********************************************************************
Closed Frequent itemset count :  15
Closed Frequent itemsets are :  ['10_1,2_4,8_4', '2_3,6_1,8_3', '10_2,2_4,7_1', '10_2,2_4,5_1', '10_2,2_4,4_1', '1_2,2_4,8_4', '1_3,2_4,5_1', '1_3,2_4,8_4', '10_2,2_4,8_4', '10_2,2_4,6_1', '10_2,2_4,3_4', '2_4,5_0,8_4', '2_4,5_1,8_3', '3_4,7_2,8_4', '3_4,6_1,8_3']
######################################################################
######################################################################
for F 5  itemset
######################################################################
number of generated candidate itemsets:  1513
number of Frequent itemsets:  316
**********************************************************************
Maximal Frequent itemset count :  78
maximal Frequent itemsets are :  [['10_1', '1_3', '4_1', '9_0'], ['10_1', '1_3', '5_1', '9_0'], ['10_1', '1_4', '5_1', '6_1'], ['10_1', '1_4', '5_1', '9_0'], ['10_1', '1_4', '6_1', '9_0'], ['10_1', '3_3', '4_1', '5_1'], ['10_1', '3_3', '5_1', '6_1'], ['10_1', '3_3', '5_1', '9_0'], ['10_1', '3_3', '6_1', '9_0'], ['10_1', '3_4', '6_1', '7_1'], ['10_1', '3_4', '8_3', '9_0'], ['10_1', '5_1', '6_1', '7_2'], ['10_1', '6_1', '7_1', '9_0'], ['10_1', '6_1', '7_2', '9_0'], ['10_2', '1_3', '3_4', '9_0'], ['10_2', '3_4', '6_1', '7_1'], ['10_2', '6_1', '7_1', '9_0'], ['10_2', '7_1', '8_4', '9_0'], ['10_3', '1_3', '3_4', '9_0'], ['10_3', '1_3', '4_1', '9_0'], ['10_3', '1_3', '5_1', '9_0'], ['10_3', '1_3', '6_1', '9_0'], ['10_3', '2_2', '5_1', '9_0'], ['10_3', '3_3', '5_1', '6_1'], ['10_3', '3_3', '5_1', '9_0'], ['10_3', '3_3', '6_1', '9_0'], ['10_3', '3_4', '7_1', '9_0'], ['10_3', '3_4', '7_3', '9_0'], ['10_3', '4_1', '7_2', '9_0'], ['10_3', '5_1', '7_2', '9_0'], ['1_2', '5_1', '6_1', '7_2'], ['1_3', '4_2', '5_1', '6_1'], ['1_3', '4_2', '5_1', '9_0'], ['1_3', '4_2', '6_1', '9_0'], ['1_3', '5_1', '6_1', '7_3'], ['1_3', '5_1', '7_3', '9_0'], ['1_3', '5_1', '8_3', '9_0'], ['1_3', '6_1', '7_3', '9_0'], ['1_4', '2_4', '3_4', '9_0'], ['1_4', '3_4', '5_1', '6_1'], ['1_4', '3_4', '5_1', '9_0'], ['1_4', '3_4', '6_1', '9_0'], ['1_4', '3_4', '8_4', '9_0'], ['1_4', '5_1', '6_1', '9_0'], ['1_4', '5_1', '8_4', '9_0'], ['1_4', '6_1', '8_4', '9_0'], ['2_2', '3_3', '5_1', '9_0'], ['2_2', '4_1', '6_1', '7_3'], ['2_3', '3_3', '4_1', '9_0'], ['2_3', '3_3', '5_1', '9_0'], ['2_3', '3_3', '6_1', '9_0'], ['2_3', '4_1', '8_3', '9_0'], ['2_3', '4_1', '8_4', '9_0'], ['2_3', '5_1', '7_2', '9_0'], ['2_3', '5_1', '8_3', '9_0'], ['2_3', '5_1', '8_4', '9_0'], ['2_3', '6_1', '8_3', '9_0'], ['2_3', '6_1', '8_4', '9_0'], ['2_4', '3_4', '4_1', '7_2'], ['2_4', '3_4', '7_2', '9_0'], ['2_4', '3_4', '7_3', '9_0'], ['2_4', '4_1', '5_1', '6_0'], ['2_4', '4_1', '7_2', '9_0'], ['2_4', '4_1', '7_3', '9_0'], ['3_3', '4_1', '6_1', '7_3'], ['3_4', '5_1', '6_0', '8_4'], ['3_4', '7_2', '8_4', '9_0'], ['3_4', '7_3', '8_4', '9_0'], ['4_1', '5_1', '6_1', '8_2'], ['4_1', '5_1', '8_2', '9_0'], ['4_1', '6_1', '8_2', '9_0'], ['4_1', '7_2', '8_4', '9_0'], ['4_2', '5_1', '8_4', '9_0'], ['4_2', '6_1', '8_4', '9_0'], ['5_1', '6_1', '8_2', '9_0'], ['5_1', '7_2', '8_4', '9_0'], ['5_1', '7_3', '8_2', '9_0'], ['6_1', '7_2', '8_4', '9_0']]
**********************************************************************
Closed Frequent itemset count :  28
Closed Frequent itemsets are :  ['1_3,2_4,4_1,6_1', '1_3,2_4,4_1,8_4', '1_3,2_4,6_1,8_4', '2_4,4_1,7_1,8_4', '1_2,2_4,8_4,9_0', '3_4,4_1,6_1,8_3', '1_3,2_4,5_1,6_1', '1_3,2_4,5_1,8_4', '2_4,3_4,5_1,8_3', '1_2,2_4,4_1,8_4', '3_4,5_1,6_1,8_3', '10_2,2_4,6_1,8_4', '10_2,2_4,3_4,8_4', '10_2,2_4,3_4,6_1', '2_4,6_1,7_1,8_4', '10_2,2_4,3_4,4_1', '2_4,3_4,5_0,8_4', '10_2,2_4,4_1,5_1', '10_1,2_4,8_4,9_0', '10_2,2_4,3_4,7_1', '10_2,2_4,3_4,5_1', '1_3,2_4,3_4,5_1', '10_2,2_4,4_1,6_1', '10_2,2_4,4_1,8_4', '1_3,2_4,4_1,5_1', '1_3,2_4,4_1,7_1', '1_3,2_4,3_4,8_4', '10_2,2_4,5_1,6_1']
######################################################################
######################################################################
for F 6  itemset
######################################################################
number of generated candidate itemsets:  419
number of Frequent itemsets:  81
**********************************************************************
Maximal Frequent itemset count :  84
maximal Frequent itemsets are :  [['10_1', '2_2', '4_1', '5_1', '9_0'], ['10_1', '2_2', '5_1', '6_1', '9_0'], ['10_1', '2_3', '4_1', '5_1', '9_0'], ['10_1', '2_3', '5_1', '6_1', '9_0'], ['10_1', '2_4', '3_4', '6_1', '9_0'], ['10_1', '2_4', '3_4', '8_4', '9_0'], ['10_1', '3_4', '4_1', '5_1', '8_4'], ['10_1', '3_4', '4_1', '7_1', '9_0'], ['10_1', '3_4', '4_1', '8_4', '9_0'], ['10_1', '3_4', '5_1', '7_1', '9_0'], ['10_1', '3_4', '5_1', '8_4', '9_0'], ['10_1', '3_4', '6_1', '8_4', '9_0'], ['10_1', '4_1', '5_1', '6_0', '9_0'], ['10_1', '4_1', '5_1', '6_1', '8_4'], ['10_1', '4_1', '5_1', '7_1', '9_0'], ['10_1', '4_1', '5_1', '7_2', '9_0'], ['10_1', '4_1', '5_1', '8_3', '9_0'], ['10_1', '4_1', '5_1', '8_4', '9_0'], ['10_1', '4_1', '6_1', '8_4', '9_0'], ['10_1', '5_1', '6_1', '8_3', '9_0'], ['10_1', '5_1', '6_1', '8_4', '9_0'], ['10_2', '2_4', '3_4', '7_1', '9_0'], ['10_2', '2_4', '6_1', '8_4', '9_0'], ['10_2', '3_4', '4_1', '7_1', '9_0'], ['10_2', '3_4', '5_1', '7_1', '9_0'], ['10_2', '3_4', '5_1', '8_4', '9_0'], ['10_2', '3_4', '6_1', '8_4', '9_0'], ['10_2', '5_1', '6_1', '8_4', '9_0'], ['10_3', '2_3', '4_1', '5_1', '9_0'], ['10_3', '2_3', '5_1', '6_1', '9_0'], ['10_3', '2_4', '3_4', '8_4', '9_0'], ['10_3', '2_4', '4_1', '8_4', '9_0'], ['10_3', '3_4', '4_1', '5_1', '8_4'], ['10_3', '3_4', '4_1', '8_4', '9_0'], ['10_3', '3_4', '5_1', '8_4', '9_0'], ['10_3', '3_4', '6_1', '8_4', '9_0'], ['10_3', '4_1', '5_1', '8_3', '9_0'], ['10_3', '4_1', '5_1', '8_4', '9_0'], ['10_3', '4_1', '6_1', '8_3', '9_0'], ['10_3', '4_1', '6_1', '8_4', '9_0'], ['10_3', '5_1', '6_1', '8_3', '9_0'], ['10_3', '5_1', '6_1', '8_4', '9_0'], ['1_2', '3_4', '4_1', '6_1', '7_3'], ['1_2', '3_4', '5_1', '6_1', '8_4'], ['1_2', '3_4', '6_1', '7_3', '9_0'], ['1_2', '4_1', '5_1', '6_0', '9_0'], ['1_2', '4_1', '5_1', '7_2', '9_0'], ['1_2', '4_1', '6_1', '7_2', '9_0'], ['1_3', '2_4', '5_1', '6_1', '9_0'], ['1_3', '3_4', '5_1', '7_1', '9_0'], ['1_3', '3_4', '6_1', '7_1', '9_0'], ['1_3', '3_4', '7_1', '8_4', '9_0'], ['1_3', '5_1', '6_1', '8_4', '9_0'], ['2_2', '4_1', '5_1', '7_3', '9_0'], ['2_2', '5_1', '6_1', '7_3', '9_0'], ['2_3', '4_1', '5_1', '7_3', '9_0'], ['2_3', '4_1', '6_1', '7_3', '9_0'], ['2_3', '5_1', '6_1', '7_3', '9_0'], ['2_4', '3_4', '4_1', '5_0', '9_0'], ['2_4', '3_4', '4_1', '6_0', '9_0'], ['2_4', '3_4', '4_1', '8_3', '9_0'], ['2_4', '3_4', '5_0', '8_4', '9_0'], ['2_4', '3_4', '5_1', '6_0', '9_0'], ['2_4', '3_4', '5_1', '8_3', '9_0'], ['2_4', '3_4', '6_0', '8_4', '9_0'], ['3_3', '4_1', '5_1', '7_3', '9_0'], ['3_3', '5_1', '6_1', '7_3', '9_0'], ['3_4', '4_1', '5_0', '6_1', '9_0'], ['3_4', '4_1', '5_0', '8_4', '9_0'], ['3_4', '4_1', '5_1', '6_0', '9_0'], ['3_4', '4_1', '5_1', '7_2', '9_0'], ['3_4', '4_1', '6_0', '8_4', '9_0'], ['3_4', '4_1', '6_1', '7_2', '9_0'], ['3_4', '4_2', '5_1', '6_1', '9_0'], ['3_4', '5_1', '6_1', '7_2', '9_0'], ['4_1', '5_1', '6_0', '8_4', '9_0'], ['4_1', '5_1', '6_1', '7_2', '9_0'], ['4_1', '5_1', '6_1', '7_3', '8_3'], ['4_1', '5_1', '7_3', '8_3', '9_0'], ['4_1', '5_1', '7_3', '8_4', '9_0'], ['4_1', '6_1', '7_3', '8_3', '9_0'], ['4_1', '6_1', '7_3', '8_4', '9_0'], ['5_1', '6_1', '7_3', '8_3', '9_0'], ['5_1', '6_1', '7_3', '8_4', '9_0']]
**********************************************************************
Closed Frequent itemset count :  18
Closed Frequent itemsets are :  ['10_2,2_4,3_4,4_1,6_1', '2_4,4_1,7_1,8_4,9_0', '1_3,2_4,3_4,5_1,8_4', '10_2,2_4,3_4,4_1,5_1', '1_3,2_4,3_4,4_1,5_1', '1_3,2_4,3_4,4_1,8_4', '2_4,3_4,6_1,7_1,8_4', '1_2,2_4,4_1,8_4,9_0', '2_4,4_1,6_1,7_1,8_4', '1_3,2_4,3_4,4_1,7_1', '3_4,4_1,5_1,6_1,8_3', '1_3,2_4,3_4,6_1,8_4', '10_2,2_4,3_4,4_1,8_4', '2_4,4_1,5_1,7_1,8_4', '10_2,2_4,4_1,8_4,9_0', '1_3,2_4,3_4,4_1,6_1', '2_4,5_1,6_1,7_1,8_4', '10_2,2_4,3_4,5_1,6_1']
######################################################################
######################################################################
for F 7  itemset
######################################################################
number of generated candidate itemsets:  38
number of Frequent itemsets:  8
**********************************************************************
Maximal Frequent itemset count :  41
maximal Frequent itemsets are :  [['10_1', '1_2', '3_4', '4_1', '5_1', '9_0'], ['10_1', '1_2', '4_1', '5_1', '6_1', '9_0'], ['10_1', '2_4', '3_4', '4_1', '5_1', '9_0'], ['10_1', '3_4', '4_1', '5_1', '6_1', '9_0'], ['10_1', '4_1', '5_1', '6_1', '7_3', '9_0'], ['10_2', '2_4', '3_4', '4_1', '5_1', '9_0'], ['10_2', '2_4', '3_4', '4_1', '6_1', '9_0'], ['10_2', '2_4', '3_4', '4_1', '8_4', '9_0'], ['10_2', '2_4', '3_4', '5_1', '6_1', '9_0'], ['10_2', '3_4', '4_1', '5_1', '6_1', '9_0'], ['10_3', '1_2', '3_4', '4_1', '5_1', '6_1'], ['10_3', '1_2', '3_4', '4_1', '5_1', '9_0'], ['10_3', '1_2', '3_4', '4_1', '6_1', '9_0'], ['10_3', '1_2', '3_4', '5_1', '6_1', '9_0'], ['10_3', '2_4', '3_4', '4_1', '5_1', '9_0'], ['10_3', '2_4', '3_4', '4_1', '6_1', '9_0'], ['10_3', '2_4', '3_4', '5_1', '6_1', '9_0'], ['10_3', '3_4', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_2', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_3', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_4', '3_4', '4_1', '8_4', '9_0'], ['1_2', '3_3', '4_1', '5_1', '6_1', '9_0'], ['1_2', '3_4', '4_1', '5_1', '7_1', '9_0'], ['1_2', '3_4', '4_1', '5_1', '7_3', '9_0'], ['1_2', '3_4', '4_1', '5_1', '8_3', '9_0'], ['1_2', '3_4', '4_1', '5_1', '8_4', '9_0'], ['1_2', '3_4', '4_1', '6_1', '8_4', '9_0'], ['1_2', '4_1', '5_1', '6_1', '8_3', '9_0'], ['1_2', '4_1', '5_1', '6_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '4_1', '5_1', '9_0'], ['1_3', '2_4', '3_4', '4_1', '6_1', '9_0'], ['1_3', '2_4', '3_4', '4_1', '7_1', '9_0'], ['1_3', '2_4', '3_4', '4_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '5_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '6_1', '8_4', '9_0'], ['1_3', '3_4', '4_1', '5_1', '6_1', '9_0'], ['1_3', '3_4', '4_1', '5_1', '8_4', '9_0'], ['1_3', '3_4', '4_1', '6_1', '8_4', '9_0'], ['2_3', '3_4', '4_1', '5_1', '6_1', '9_0'], ['3_4', '4_1', '5_1', '6_1', '7_3', '9_0'], ['3_4', '4_1', '5_1', '6_1', '8_3', '9_0']]
**********************************************************************
Closed Frequent itemset count :  4
Closed Frequent itemsets are :  ['2_4,4_1,6_1,7_1,8_4,9_0', '2_4,4_1,5_1,7_1,8_4,9_0', '2_4,3_4,4_1,6_1,7_1,8_4', '2_4,3_4,5_1,6_1,7_1,8_4']
######################################################################


C:\Users\samee\OneDrive\Assignments\DM assignent 4\Question3>python apriori_c_contraceptive.py
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\contraceptive.csv
Enter Minimum support %: 5
minimum support count :  73
columns decoded as ColumnIndex_value
column names are as below
['10_1', '10_2', '10_3', '1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '5_0', '5_1', '6_0', '6_1', '7_1', '7_2', '7_3', '7_4', '8_1', '8_2', '8_3', '8_4', '9_0', '9_1']
*********************
Using method Fk-1*F1
*********************
######################################################################
for F1 itemset
######################################################################
number of generated candidate itemsets:  33
number of Frequent itemsets:  28
######################################################################
######################################################################
for F2 itemset
######################################################################
number of generated candidate itemsets:  350
number of Frequent itemsets:  249
**********************************************************************
Maximal Frequent itemset count :  0
maximal Frequent itemsets are :  []
**********************************************************************
Closed Frequent itemset count :  0
Closed Frequent itemsets are :  []
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  1910
number of Frequent itemsets:  729
**********************************************************************
Maximal Frequent itemset count :  6
maximal Frequent itemsets are :  [['10_1', '5_0'], ['10_1', '9_1'], ['1_3', '8_2'], ['2_2', '6_0'], ['7_2', '8_2'], ['7_3', '8_1']]
**********************************************************************
Closed Frequent itemset count :  2
Closed Frequent itemsets are :  ['2_1,7_3', '10_2,2_4']
######################################################################
######################################################################
for F 4  itemset
######################################################################
number of generated candidate itemsets:  3501
number of Frequent itemsets:  996
**********************************************************************
Maximal Frequent itemset count :  46
maximal Frequent itemsets are :  [['10_1', '1_3', '7_3'], ['10_1', '1_4', '8_4'], ['10_1', '2_3', '7_3'], ['10_1', '5_1', '8_1'], ['10_1', '7_3', '8_3'], ['10_1', '7_3', '8_4'], ['10_2', '1_2', '6_1'], ['10_2', '1_4', '9_0'], ['10_2', '2_3', '9_0'], ['10_2', '4_2', '5_1'], ['10_2', '4_2', '9_0'], ['10_2', '5_0', '9_0'], ['10_2', '5_1', '8_3'], ['10_2', '6_0', '9_0'], ['10_2', '7_2', '9_0'], ['10_2', '8_3', '9_0'], ['10_3', '5_1', '8_2'], ['1_2', '5_1', '8_1'], ['1_3', '2_3', '4_1'], ['1_3', '3_4', '8_3'], ['1_4', '5_1', '7_3'], ['1_4', '5_1', '8_3'], ['1_4', '7_2', '9_0'], ['2_1', '4_1', '5_1'], ['2_1', '5_1', '7_3'], ['2_1', '5_1', '9_0'], ['2_2', '3_2', '5_1'], ['2_2', '3_2', '9_0'], ['2_2', '3_3', '7_3'], ['2_2', '4_1', '8_4'], ['2_3', '4_2', '5_1'], ['2_3', '4_2', '9_0'], ['2_3', '5_1', '7_1'], ['2_3', '5_1', '8_2'], ['2_3', '7_1', '9_0'], ['3_3', '5_1', '6_0'], ['3_3', '5_1', '8_2'], ['3_3', '6_0', '9_0'], ['3_4', '4_2', '7_1'], ['3_4', '5_1', '8_2'], ['3_4', '8_2', '9_0'], ['4_2', '5_1', '7_2'], ['4_2', '6_1', '7_1'], ['4_2', '6_1', '8_3'], ['5_0', '7_2', '9_0'], ['5_1', '6_1', '9_1']]
**********************************************************************
Closed Frequent itemset count :  30
Closed Frequent itemsets are :  ['1_2,2_2,7_3', '10_1,2_4,8_4', '10_1,2_4,7_1', '4_1,7_1,8_3', '10_2,2_4,8_4', '2_4,7_2,8_4', '10_2,1_3,2_4', '2_4,4_2,5_1', '5_0,6_1,8_4', '2_4,6_1,8_3', '6_1,7_1,8_3', '10_3,2_3,3_4', '10_1,1_2,2_2', '1_2,2_4,7_1', '10_2,2_4,7_1', '10_2,2_4,5_1', '2_3,6_1,8_3', '1_2,2_3,3_3', '10_2,2_4,4_1', '1_2,2_4,8_3', '1_2,2_4,8_4', '1_3,2_4,5_1', '1_3,2_4,8_4', '10_2,2_4,6_1', '1_4,2_4,6_1', '10_2,2_4,3_4', '2_4,5_0,8_4', '2_4,5_1,8_3', '3_4,7_2,8_4', '3_4,6_1,8_3']
######################################################################
######################################################################
for F 5  itemset
######################################################################
number of generated candidate itemsets:  2885
number of Frequent itemsets:  692
**********************************************************************
Maximal Frequent itemset count :  96
maximal Frequent itemsets are :  [['10_1', '1_3', '3_4', '9_0'], ['10_1', '1_3', '4_1', '6_1'], ['10_1', '1_4', '3_4', '9_0'], ['10_1', '1_4', '4_1', '5_1'], ['10_1', '1_4', '4_1', '9_0'], ['10_1', '2_1', '5_1', '6_1'], ['10_1', '2_2', '4_1', '7_3'], ['10_1', '3_2', '5_1', '6_1'], ['10_1', '3_3', '5_1', '7_3'], ['10_1', '3_4', '7_2', '9_0'], ['10_1', '4_1', '5_1', '8_2'], ['10_1', '4_1', '7_1', '8_4'], ['10_1', '5_1', '6_1', '8_2'], ['10_1', '5_1', '8_2', '9_0'], ['10_2', '1_2', '3_4', '4_1'], ['10_2', '1_2', '3_4', '9_0'], ['10_2', '5_1', '7_3', '9_0'], ['10_3', '1_2', '2_4', '3_4'], ['10_3', '1_2', '7_2', '9_0'], ['10_3', '1_3', '2_4', '3_4'], ['10_3', '1_3', '2_4', '9_0'], ['10_3', '1_3', '4_1', '6_1'], ['10_3', '2_2', '4_1', '5_1'], ['10_3', '2_2', '4_1', '9_0'], ['10_3', '2_3', '3_4', '9_0'], ['10_3', '2_3', '7_3', '9_0'], ['10_3', '4_1', '6_0', '9_0'], ['10_3', '4_2', '5_1', '6_1'], ['10_3', '4_2', '5_1', '9_0'], ['10_3', '4_2', '6_1', '9_0'], ['10_3', '5_1', '6_0', '9_0'], ['1_2', '2_3', '3_3', '9_0'], ['1_2', '4_1', '7_1', '8_4'], ['1_2', '4_1', '7_3', '8_4'], ['1_2', '5_1', '6_1', '8_2'], ['1_3', '2_2', '5_1', '6_1'], ['1_3', '2_2', '5_1', '9_0'], ['1_3', '2_3', '5_1', '9_0'], ['1_3', '2_3', '6_1', '9_0'], ['1_3', '3_3', '5_1', '9_0'], ['1_3', '3_4', '7_3', '9_0'], ['1_3', '4_1', '5_1', '7_3'], ['1_3', '4_1', '6_1', '7_3'], ['1_3', '4_1', '7_2', '9_0'], ['1_3', '4_1', '7_3', '9_0'], ['1_3', '4_1', '8_3', '9_0'], ['1_3', '5_0', '8_4', '9_0'], ['1_3', '5_1', '6_0', '9_0'], ['1_3', '5_1', '6_1', '7_2'], ['1_3', '5_1', '7_2', '9_0'], ['1_3', '6_0', '8_4', '9_0'], ['1_3', '6_1', '7_2', '9_0'], ['1_4', '3_4', '4_1', '9_0'], ['1_4', '4_1', '5_1', '6_1'], ['1_4', '4_1', '5_1', '9_0'], ['1_4', '4_1', '6_1', '9_0'], ['1_4', '4_1', '8_4', '9_0'], ['1_4', '4_2', '5_1', '6_1'], ['1_4', '4_2', '5_1', '9_0'], ['1_4', '4_2', '6_1', '9_0'], ['1_4', '5_1', '7_1', '8_4'], ['2_2', '4_2', '5_1', '9_0'], ['2_2', '5_1', '6_1', '7_2'], ['2_2', '5_1', '6_1', '8_4'], ['2_2', '5_1', '7_2', '9_0'], ['2_2', '5_1', '8_4', '9_0'], ['2_2', '6_1', '7_2', '9_0'], ['2_2', '6_1', '8_4', '9_0'], ['2_3', '3_3', '7_3', '9_0'], ['2_3', '3_4', '5_1', '8_3'], ['2_3', '3_4', '8_3', '9_0'], ['2_3', '3_4', '8_4', '9_0'], ['2_3', '4_1', '6_1', '8_4'], ['2_3', '5_1', '6_0', '9_0'], ['2_4', '3_4', '4_2', '9_0'], ['2_4', '4_2', '5_1', '9_0'], ['3_2', '4_1', '5_1', '6_1'], ['3_2', '4_1', '5_1', '9_0'], ['3_2', '5_1', '6_1', '7_3'], ['3_2', '5_1', '6_1', '9_0'], ['3_2', '5_1', '7_3', '9_0'], ['3_3', '4_2', '5_1', '9_0'], ['3_3', '5_1', '8_4', '9_0'], ['3_3', '6_1', '8_4', '9_0'], ['4_1', '5_1', '6_1', '8_1'], ['4_1', '5_1', '8_1', '9_0'], ['4_1', '6_0', '7_2', '9_0'], ['4_1', '6_0', '8_3', '9_0'], ['4_1', '7_2', '8_3', '9_0'], ['4_2', '5_1', '7_1', '9_0'], ['4_2', '5_1', '8_3', '9_0'], ['4_2', '6_1', '7_2', '9_0'], ['5_1', '6_0', '7_2', '9_0'], ['5_1', '6_0', '8_3', '9_0'], ['5_1', '6_1', '8_1', '9_0'], ['6_1', '7_1', '8_3', '9_0']]
**********************************************************************
Closed Frequent itemset count :  80
Closed Frequent itemsets are :  ['10_3,3_4,6_1,8_3', '1_3,2_4,4_1,6_1', '1_3,2_4,4_1,8_4', '1_3,2_4,7_1,8_4', '2_4,3_4,6_1,8_3', '10_2,2_4,7_1,8_4', '1_3,2_4,3_4,8_4', '1_2,3_4,6_1,8_3', '1_3,2_4,6_1,8_4', '10_2,2_4,5_1,8_4', '1_2,2_4,4_1,7_1', '2_3,4_1,5_1,8_3', '2_4,4_1,7_1,8_4', '1_2,2_4,8_4,9_0', '2_4,4_1,5_1,8_3', '1_2,2_4,7_1,9_0', '10_2,2_4,5_1,7_1', '3_4,6_1,7_2,8_4', '1_3,2_4,3_4,5_1', '2_4,3_4,5_0,8_4', '3_4,4_1,7_1,8_3', '10_2,2_4,6_1,7_1', '10_1,1_2,2_2,4_1', '2_4,4_1,6_0,8_4', '3_4,4_1,6_1,8_3', '2_3,4_1,6_1,8_3', '1_3,2_4,5_1,6_1', '1_3,2_4,5_1,8_4', '2_4,3_4,5_1,8_3', '1_2,2_4,4_1,8_3', '1_2,2_4,4_1,8_4', '10_1,3_4,6_1,8_3', '2_4,4_1,6_0,7_1', '10_1,2_4,5_1,8_4', '1_2,2_2,4_1,7_3', '10_2,1_3,2_4,3_4', '1_2,2_4,8_3,9_0', '2_4,4_1,7_2,8_4', '10_1,2_4,5_1,7_1', '1_2,2_4,5_1,7_1', '1_2,2_2,6_1,7_3', '10_2,2_4,6_1,8_4', '4_1,5_1,7_1,8_3', '10_2,2_4,3_4,8_4', '1_3,2_4,5_1,7_1', '10_2,1_3,3_4,6_1', '2_4,4_1,6_1,8_3', '4_1,5_0,6_1,8_4', '10_2,2_4,3_4,6_1', '2_4,6_1,7_1,8_4', '10_2,2_4,3_4,4_1', '3_4,6_1,7_3,8_3', '2_4,3_4,7_2,8_4', '10_2,2_4,4_1,7_1', '10_1,2_4,4_1,7_1', '10_2,2_4,4_1,5_1', '3_4,4_1,7_2,8_4', '1_4,2_4,3_4,6_1', '3_4,5_1,6_1,8_3', '10_1,2_4,8_4,9_0', '2_3,5_1,6_1,8_3', '2_4,4_1,5_0,8_4', '10_2,2_4,3_4,7_1', '2_4,4_1,5_0,6_1', '10_3,3_4,5_1,8_3', '10_2,2_4,3_4,5_1', '10_1,2_4,4_1,8_4', '10_2,2_4,4_1,6_1', '1_4,2_4,6_1,8_4', '10_2,2_4,4_1,8_4', '1_3,2_4,4_1,5_1', '2_4,5_1,6_1,8_3', '1_3,4_1,7_1,8_4', '1_2,2_2,7_3,9_0', '3_4,5_0,6_1,8_4', '1_3,2_4,4_1,7_1', '10_2,2_4,5_1,6_1', '10_1,2_4,7_1,9_0', '1_2,7_1,8_4,9_0', '1_2,2_4,5_1,8_4']
######################################################################
######################################################################
for F 6  itemset
######################################################################
number of generated candidate itemsets:  998
number of Frequent itemsets:  225
**********************************************************************
Maximal Frequent itemset count :  147
maximal Frequent itemsets are :  [['10_1', '1_2', '2_2', '4_1', '5_1'], ['10_1', '1_2', '2_4', '3_4', '9_0'], ['10_1', '1_2', '2_4', '4_1', '9_0'], ['10_1', '1_2', '4_1', '6_1', '7_3'], ['10_1', '1_2', '5_1', '6_1', '7_3'], ['10_1', '1_3', '4_1', '5_1', '9_0'], ['10_1', '1_3', '5_1', '6_1', '9_0'], ['10_1', '1_4', '5_1', '6_1', '9_0'], ['10_1', '2_2', '5_1', '7_3', '9_0'], ['10_1', '2_3', '3_4', '4_1', '9_0'], ['10_1', '2_3', '3_4', '5_1', '9_0'], ['10_1', '2_3', '3_4', '6_1', '9_0'], ['10_1', '2_4', '4_1', '5_1', '6_1'], ['10_1', '3_3', '4_1', '5_1', '6_1'], ['10_1', '3_3', '4_1', '5_1', '9_0'], ['10_1', '3_3', '4_1', '6_1', '9_0'], ['10_1', '3_3', '5_1', '6_1', '9_0'], ['10_1', '3_4', '4_1', '6_0', '9_0'], ['10_1', '3_4', '5_1', '6_0', '9_0'], ['10_1', '3_4', '6_1', '7_3', '9_0'], ['10_1', '3_4', '6_1', '8_3', '9_0'], ['10_1', '3_4', '7_1', '8_4', '9_0'], ['10_1', '4_1', '5_1', '6_0', '9_0'], ['10_1', '4_1', '5_1', '6_1', '7_1'], ['10_1', '4_1', '5_1', '6_1', '7_2'], ['10_1', '4_1', '5_1', '7_2', '9_0'], ['10_1', '4_1', '6_1', '7_2', '9_0'], ['10_1', '4_2', '5_1', '6_1', '9_0'], ['10_1', '5_1', '6_1', '7_2', '9_0'], ['10_1', '5_1', '7_1', '8_4', '9_0'], ['10_2', '1_2', '4_1', '5_1', '9_0'], ['10_2', '1_3', '2_4', '3_4', '9_0'], ['10_2', '1_3', '3_4', '4_1', '9_0'], ['10_2', '1_3', '3_4', '6_1', '9_0'], ['10_2', '1_3', '3_4', '8_4', '9_0'], ['10_2', '1_3', '5_1', '6_1', '9_0'], ['10_3', '1_2', '2_3', '5_1', '6_1'], ['10_3', '1_2', '2_3', '6_1', '9_0'], ['10_3', '1_2', '2_4', '4_1', '9_0'], ['10_3', '1_2', '3_3', '5_1', '9_0'], ['10_3', '1_2', '4_1', '8_4', '9_0'], ['10_3', '1_2', '5_1', '8_4', '9_0'], ['10_3', '1_3', '3_4', '4_1', '9_0'], ['10_3', '1_3', '3_4', '5_1', '9_0'], ['10_3', '1_3', '3_4', '6_1', '9_0'], ['10_3', '1_3', '3_4', '8_4', '9_0'], ['10_3', '1_3', '4_1', '5_1', '9_0'], ['10_3', '1_3', '4_1', '8_4', '9_0'], ['10_3', '1_3', '5_1', '6_1', '9_0'], ['10_3', '2_2', '5_1', '6_1', '9_0'], ['10_3', '2_4', '3_4', '7_1', '9_0'], ['10_3', '2_4', '4_1', '5_1', '8_4'], ['10_3', '2_4', '4_1', '6_1', '8_4'], ['10_3', '3_3', '4_1', '5_1', '6_1'], ['10_3', '3_3', '4_1', '5_1', '9_0'], ['10_3', '3_3', '4_1', '6_1', '9_0'], ['10_3', '3_3', '5_1', '6_1', '9_0'], ['10_3', '3_3', '5_1', '7_3', '9_0'], ['10_3', '3_4', '4_1', '7_1', '9_0'], ['10_3', '3_4', '4_1', '8_3', '9_0'], ['10_3', '3_4', '5_1', '7_1', '9_0'], ['10_3', '3_4', '5_1', '8_3', '9_0'], ['10_3', '3_4', '6_1', '7_1', '9_0'], ['10_3', '3_4', '6_1', '8_3', '9_0'], ['10_3', '3_4', '7_1', '8_4', '9_0'], ['10_3', '4_1', '5_1', '7_2', '9_0'], ['10_3', '4_1', '6_1', '7_2', '9_0'], ['10_3', '5_1', '6_1', '7_1', '9_0'], ['10_3', '5_1', '6_1', '7_2', '9_0'], ['10_3', '5_1', '7_1', '8_4', '9_0'], ['1_2', '2_3', '3_4', '5_1', '6_1'], ['1_2', '2_3', '4_1', '6_1', '7_3'], ['1_2', '2_3', '5_1', '6_1', '7_3'], ['1_2', '2_3', '6_1', '7_3', '9_0'], ['1_2', '2_4', '4_1', '8_3', '9_0'], ['1_2', '3_4', '4_1', '7_2', '9_0'], ['1_2', '3_4', '7_1', '8_4', '9_0'], ['1_2', '4_1', '5_1', '8_2', '9_0'], ['1_2', '4_1', '6_1', '7_3', '8_3'], ['1_2', '5_1', '7_3', '8_4', '9_0'], ['1_3', '3_4', '4_1', '5_0', '9_0'], ['1_3', '3_4', '4_1', '6_0', '9_0'], ['1_3', '4_2', '5_1', '6_1', '9_0'], ['1_3', '5_1', '6_1', '7_3', '9_0'], ['1_3', '5_1', '6_1', '8_3', '9_0'], ['1_4', '2_4', '3_4', '5_1', '9_0'], ['1_4', '2_4', '3_4', '7_1', '9_0'], ['1_4', '2_4', '7_1', '8_4', '9_0'], ['1_4', '3_4', '5_1', '6_1', '9_0'], ['1_4', '3_4', '5_1', '7_1', '9_0'], ['1_4', '3_4', '5_1', '8_4', '9_0'], ['1_4', '3_4', '6_1', '7_1', '9_0'], ['1_4', '3_4', '7_1', '8_4', '9_0'], ['1_4', '5_1', '6_1', '7_1', '9_0'], ['1_4', '5_1', '6_1', '8_4', '9_0'], ['2_2', '3_3', '4_1', '5_1', '9_0'], ['2_2', '3_3', '5_1', '6_1', '9_0'], ['2_2', '3_4', '4_1', '5_1', '9_0'], ['2_2', '3_4', '5_1', '6_1', '9_0'], ['2_2', '5_1', '6_1', '8_3', '9_0'], ['2_3', '3_3', '4_1', '5_1', '9_0'], ['2_3', '3_3', '4_1', '6_1', '9_0'], ['2_3', '3_3', '5_1', '6_1', '9_0'], ['2_3', '3_4', '4_1', '7_3', '9_0'], ['2_3', '3_4', '5_1', '7_3', '9_0'], ['2_3', '3_4', '6_1', '7_3', '9_0'], ['2_3', '4_1', '5_1', '7_2', '9_0'], ['2_3', '4_1', '5_1', '8_4', '9_0'], ['2_3', '4_1', '6_1', '7_2', '9_0'], ['2_3', '5_1', '6_1', '7_2', '9_0'], ['2_3', '5_1', '6_1', '8_4', '9_0'], ['2_4', '3_4', '4_1', '5_1', '7_3'], ['2_4', '3_4', '4_1', '6_1', '7_3'], ['2_4', '3_4', '4_1', '7_3', '9_0'], ['2_4', '3_4', '5_1', '7_2', '9_0'], ['2_4', '3_4', '5_1', '7_3', '9_0'], ['2_4', '3_4', '6_1', '7_2', '9_0'], ['2_4', '3_4', '6_1', '7_3', '9_0'], ['2_4', '4_1', '5_1', '7_3', '9_0'], ['2_4', '4_1', '6_1', '7_3', '9_0'], ['2_4', '5_1', '6_1', '7_3', '9_0'], ['3_3', '4_1', '5_1', '7_2', '9_0'], ['3_3', '5_1', '6_1', '7_2', '9_0'], ['3_3', '5_1', '6_1', '8_3', '9_0'], ['3_4', '4_1', '7_1', '8_3', '9_0'], ['3_4', '4_1', '7_3', '8_4', '9_0'], ['3_4', '4_2', '5_1', '6_1', '9_0'], ['3_4', '4_2', '5_1', '8_4', '9_0'], ['3_4', '4_2', '6_1', '8_4', '9_0'], ['3_4', '5_1', '6_0', '7_1', '9_0'], ['3_4', '5_1', '7_1', '8_3', '9_0'], ['3_4', '5_1', '7_3', '8_4', '9_0'], ['3_4', '6_0', '7_1', '8_4', '9_0'], ['3_4', '6_1', '7_2', '8_4', '9_0'], ['3_4', '6_1', '7_3', '8_3', '9_0'], ['3_4', '6_1', '7_3', '8_4', '9_0'], ['4_1', '5_1', '6_0', '7_3', '9_0'], ['4_1', '5_1', '6_1', '8_2', '9_0'], ['4_1', '5_1', '7_1', '8_3', '9_0'], ['4_1', '5_1', '7_2', '8_4', '9_0'], ['4_1', '5_1', '7_3', '8_2', '9_0'], ['4_1', '6_1', '7_2', '8_4', '9_0'], ['4_2', '5_1', '6_1', '7_3', '9_0'], ['4_2', '5_1', '6_1', '8_4', '9_0'], ['5_1', '6_1', '7_2', '8_3', '9_0'], ['5_1', '6_1', '7_2', '8_4', '9_0'], ['5_1', '6_1', '7_3', '8_2', '9_0']]
**********************************************************************
Closed Frequent itemset count :  63
Closed Frequent itemsets are :  ['10_2,2_4,3_4,4_1,6_1', '1_3,2_4,4_1,5_1,7_1', '2_4,3_4,4_1,7_2,8_4', '1_2,2_4,5_1,7_1,9_0', '10_2,2_4,3_4,5_1,7_1', '2_4,4_1,7_1,8_4,9_0', '10_2,2_4,5_1,6_1,8_4', '1_3,2_4,3_4,5_1,8_4', '1_4,2_4,3_4,6_1,8_4', '2_4,4_1,6_1,7_1,8_4', '1_2,3_4,5_1,6_1,8_3', '1_3,2_4,3_4,5_1,7_1', '1_2,2_2,4_1,7_3,9_0', '1_3,2_4,5_1,6_1,8_4', '1_3,2_4,3_4,4_1,7_1', '1_3,2_4,3_4,4_1,6_1', '2_4,5_1,6_1,7_1,8_4', '10_2,2_4,3_4,5_1,6_1', '10_1,2_4,5_1,8_4,9_0', '10_2,2_4,3_4,5_1,8_4', '2_4,4_1,6_0,7_1,9_0', '1_3,2_4,3_4,4_1,8_4', '2_4,3_4,4_1,5_1,8_3', '10_2,2_4,4_1,6_1,8_4', '10_2,2_4,3_4,6_1,7_1', '10_2,2_4,4_1,5_1,6_1', '1_2,3_4,4_1,6_1,8_3', '1_3,2_4,3_4,4_1,5_1', '2_4,3_4,6_1,7_1,8_4', '1_2,2_4,4_1,7_1,9_0', '2_4,3_4,4_1,5_0,8_4', '2_4,4_1,5_1,7_1,8_4', '1_3,2_4,5_1,7_1,8_4', '10_2,2_4,3_4,6_1,8_4', '1_3,2_4,3_4,7_1,8_4', '2_4,4_1,6_0,8_4,9_0', '1_2,2_4,4_1,5_1,8_4', '2_3,4_1,5_1,6_1,8_3', '1_3,2_4,3_4,6_1,8_4', '3_4,4_1,5_1,6_1,8_3', '10_1,2_4,4_1,7_1,9_0', '2_4,3_4,4_1,5_0,6_1', '2_4,3_4,4_1,6_1,8_3', '10_2,2_4,4_1,8_4,9_0', '1_3,3_4,4_1,7_1,8_4', '1_3,4_1,5_1,7_1,8_4', '1_3,2_4,4_1,6_1,8_4', '3_4,4_1,5_0,6_1,8_4', '1_3,2_4,4_1,5_1,8_4', '10_2,2_4,3_4,4_1,7_1', '1_3,2_4,4_1,5_1,6_1', '10_2,2_4,3_4,4_1,5_1', '10_2,2_4,3_4,7_1,8_4', '2_4,3_4,5_1,6_1,8_3', '10_2,2_4,3_4,4_1,8_4', '1_2,2_4,4_1,5_1,7_1', '1_2,2_4,5_1,8_4,9_0', '1_2,2_4,4_1,8_4,9_0', '1_3,2_4,4_1,7_1,8_4', '1_3,2_4,3_4,5_1,6_1', '1_2,2_2,6_1,7_3,9_0', '10_1,2_4,4_1,8_4,9_0', '10_1,2_4,5_1,7_1,9_0']
######################################################################
######################################################################
for F 7  itemset
######################################################################
number of generated candidate itemsets:  110
number of Frequent itemsets:  30
**********************************************************************
Maximal Frequent itemset count :  89
maximal Frequent itemsets are :  [['10_1', '1_2', '4_1', '5_1', '7_3', '9_0'], ['10_1', '1_2', '4_1', '5_1', '8_4', '9_0'], ['10_1', '2_2', '4_1', '5_1', '6_1', '9_0'], ['10_1', '2_3', '4_1', '5_1', '6_1', '9_0'], ['10_1', '2_4', '3_4', '4_1', '5_1', '9_0'], ['10_1', '2_4', '3_4', '4_1', '6_1', '9_0'], ['10_1', '2_4', '3_4', '4_1', '7_1', '9_0'], ['10_1', '2_4', '3_4', '4_1', '8_4', '9_0'], ['10_1', '2_4', '3_4', '5_1', '6_1', '9_0'], ['10_1', '2_4', '3_4', '5_1', '7_1', '9_0'], ['10_1', '2_4', '3_4', '5_1', '8_4', '9_0'], ['10_1', '3_4', '4_1', '5_1', '7_1', '9_0'], ['10_1', '3_4', '4_1', '5_1', '7_3', '9_0'], ['10_1', '3_4', '4_1', '5_1', '8_3', '9_0'], ['10_1', '3_4', '4_1', '5_1', '8_4', '9_0'], ['10_1', '3_4', '4_1', '6_1', '7_1', '9_0'], ['10_1', '3_4', '4_1', '6_1', '8_4', '9_0'], ['10_1', '3_4', '5_1', '6_1', '7_1', '9_0'], ['10_1', '3_4', '5_1', '6_1', '8_4', '9_0'], ['10_1', '4_1', '5_1', '6_1', '7_3', '9_0'], ['10_1', '4_1', '5_1', '6_1', '8_3', '9_0'], ['10_1', '4_1', '5_1', '6_1', '8_4', '9_0'], ['10_2', '2_4', '3_4', '4_1', '7_1', '9_0'], ['10_2', '2_4', '3_4', '5_1', '7_1', '9_0'], ['10_2', '2_4', '3_4', '6_1', '7_1', '9_0'], ['10_2', '2_4', '3_4', '7_1', '8_4', '9_0'], ['10_2', '3_4', '4_1', '5_1', '7_1', '9_0'], ['10_2', '3_4', '4_1', '5_1', '8_4', '9_0'], ['10_2', '3_4', '4_1', '6_1', '7_1', '9_0'], ['10_2', '3_4', '4_1', '7_1', '8_4', '9_0'], ['10_2', '3_4', '5_1', '6_1', '7_1', '9_0'], ['10_2', '3_4', '5_1', '7_1', '8_4', '9_0'], ['10_2', '3_4', '6_1', '7_1', '8_4', '9_0'], ['10_3', '1_2', '2_3', '4_1', '5_1', '9_0'], ['10_3', '2_3', '4_1', '5_1', '6_1', '9_0'], ['10_3', '2_4', '3_4', '4_1', '8_4', '9_0'], ['10_3', '2_4', '3_4', '5_1', '8_4', '9_0'], ['10_3', '2_4', '3_4', '6_1', '8_4', '9_0'], ['10_3', '3_4', '4_1', '5_1', '6_1', '8_4'], ['10_3', '3_4', '4_1', '5_1', '7_3', '9_0'], ['10_3', '3_4', '4_1', '5_1', '8_4', '9_0'], ['10_3', '3_4', '4_1', '6_1', '7_3', '9_0'], ['10_3', '3_4', '4_1', '6_1', '8_4', '9_0'], ['10_3', '3_4', '5_1', '6_1', '7_3', '9_0'], ['10_3', '3_4', '5_1', '6_1', '8_4', '9_0'], ['10_3', '4_1', '5_1', '6_1', '8_4', '9_0'], ['10_3', '5_1', '6_1', '7_3', '8_3', '9_0'], ['1_2', '2_2', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_2', '4_1', '5_1', '7_3', '9_0'], ['1_2', '2_2', '5_1', '6_1', '7_3', '9_0'], ['1_2', '2_3', '3_4', '4_1', '5_1', '9_0'], ['1_2', '2_3', '3_4', '4_1', '6_1', '9_0'], ['1_2', '2_3', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_3', '4_1', '5_1', '7_3', '9_0'], ['1_2', '3_3', '4_1', '5_1', '6_1', '9_0'], ['1_2', '3_3', '4_1', '5_1', '7_3', '9_0'], ['1_2', '3_3', '5_1', '6_1', '7_3', '9_0'], ['1_2', '3_4', '4_1', '5_1', '6_0', '9_0'], ['1_2', '4_1', '5_1', '6_1', '7_2', '9_0'], ['1_2', '4_1', '5_1', '7_3', '8_3', '9_0'], ['1_2', '5_1', '6_1', '7_3', '8_3', '9_0'], ['1_3', '2_4', '3_4', '6_1', '7_1', '9_0'], ['1_3', '3_4', '4_1', '6_1', '7_1', '9_0'], ['1_3', '3_4', '5_1', '6_1', '7_1', '9_0'], ['1_3', '3_4', '6_1', '7_1', '8_4', '9_0'], ['1_3', '4_1', '5_1', '6_1', '8_4', '9_0'], ['1_3', '4_1', '5_1', '7_1', '8_4', '9_0'], ['1_4', '2_4', '3_4', '6_1', '8_4', '9_0'], ['2_2', '4_1', '5_1', '6_1', '7_3', '9_0'], ['2_3', '3_4', '4_1', '5_1', '6_1', '9_0'], ['2_3', '4_1', '5_1', '6_1', '7_3', '9_0'], ['2_3', '4_1', '5_1', '6_1', '8_3', '9_0'], ['2_4', '3_4', '4_1', '5_0', '6_1', '9_0'], ['2_4', '3_4', '4_1', '5_0', '8_4', '9_0'], ['2_4', '3_4', '4_1', '5_1', '6_0', '9_0'], ['2_4', '3_4', '4_1', '5_1', '8_3', '9_0'], ['2_4', '3_4', '4_1', '6_0', '7_1', '9_0'], ['2_4', '3_4', '4_1', '6_0', '8_4', '9_0'], ['2_4', '3_4', '4_1', '6_1', '8_3', '9_0'], ['2_4', '3_4', '4_1', '7_2', '8_4', '9_0'], ['2_4', '3_4', '5_1', '6_0', '8_4', '9_0'], ['2_4', '3_4', '5_1', '6_1', '8_3', '9_0'], ['3_3', '4_1', '5_1', '6_1', '7_3', '9_0'], ['3_4', '4_1', '5_0', '6_1', '8_4', '9_0'], ['3_4', '4_1', '5_1', '6_0', '8_4', '9_0'], ['3_4', '4_1', '5_1', '6_1', '7_2', '9_0'], ['3_4', '4_1', '5_1', '7_3', '8_3', '9_0'], ['4_1', '5_1', '6_1', '7_3', '8_3', '9_0'], ['4_1', '5_1', '6_1', '7_3', '8_4', '9_0']]
**********************************************************************
Closed Frequent itemset count :  21
Closed Frequent itemsets are :  ['2_4,4_1,6_1,7_1,8_4,9_0', '1_3,2_4,3_4,4_1,5_1,7_1', '1_3,2_4,3_4,5_1,7_1,8_4', '1_3,2_4,3_4,4_1,5_1,6_1', '1_3,2_4,4_1,7_1,8_4,9_0', '2_4,3_4,5_1,6_1,7_1,8_4', '1_2,3_4,4_1,5_1,6_1,8_3', '2_4,4_1,5_1,7_1,8_4,9_0', '1_3,2_4,3_4,5_1,6_1,8_4', '1_3,2_4,3_4,4_1,5_1,8_4', '1_2,2_4,4_1,5_1,7_1,9_0', '1_3,2_4,4_1,5_1,7_1,9_0', '1_3,2_4,3_4,4_1,6_1,8_4', '1_3,2_4,3_4,4_1,7_1,8_4', '10_2,2_4,4_1,6_1,8_4,9_0', '2_4,3_4,4_1,6_1,7_1,8_4', '10_2,2_4,3_4,4_1,6_1,8_4', '1_2,2_4,4_1,5_1,8_4,9_0', '10_2,2_4,3_4,4_1,5_1,6_1', '10_2,2_4,3_4,5_1,6_1,8_4', '2_4,4_1,5_1,6_1,7_1,8_4']
######################################################################
######################################################################
for F 8  itemset
######################################################################
number of generated candidate itemsets:  1
number of Frequent itemsets:  1
**********************************************************************
Maximal Frequent itemset count :  22
maximal Frequent itemsets are :  [['10_1', '1_2', '3_4', '4_1', '5_1', '6_1', '9_0'], ['10_2', '2_4', '3_4', '4_1', '5_1', '6_1', '9_0'], ['10_2', '2_4', '3_4', '4_1', '6_1', '8_4', '9_0'], ['10_2', '2_4', '3_4', '5_1', '6_1', '8_4', '9_0'], ['10_3', '1_2', '3_4', '4_1', '5_1', '6_1', '9_0'], ['10_3', '1_2', '4_1', '5_1', '6_1', '7_3', '9_0'], ['10_3', '1_2', '4_1', '5_1', '6_1', '8_3', '9_0'], ['10_3', '2_4', '3_4', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_4', '3_4', '4_1', '5_1', '6_1', '9_0'], ['1_2', '2_4', '3_4', '4_1', '5_1', '7_1', '9_0'], ['1_2', '2_4', '3_4', '4_1', '5_1', '8_4', '9_0'], ['1_2', '3_4', '4_1', '5_1', '6_1', '7_1', '9_0'], ['1_2', '3_4', '4_1', '5_1', '6_1', '7_3', '9_0'], ['1_2', '3_4', '4_1', '5_1', '6_1', '8_3', '9_0'], ['1_2', '3_4', '4_1', '5_1', '6_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '4_1', '5_1', '6_1', '9_0'], ['1_3', '2_4', '3_4', '4_1', '5_1', '7_1', '9_0'], ['1_3', '2_4', '3_4', '4_1', '5_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '4_1', '6_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '4_1', '7_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '5_1', '6_1', '8_4', '9_0'], ['1_3', '2_4', '3_4', '5_1', '7_1', '8_4', '9_0']]
**********************************************************************
Closed Frequent itemset count :  2
Closed Frequent itemsets are :  ['2_4,3_4,4_1,5_1,6_1,7_1,8_4', '2_4,4_1,5_1,6_1,7_1,8_4,9_0']
######################################################################


"""