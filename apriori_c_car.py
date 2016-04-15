# Author: Sameer Darekar
# Title: Implementing Apriori Algorithm
# Dataset Location: https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data

"""
   Attribute Information
   buying       v-high, high, med, low
   maint        v-high, high, med, low
   doors        2, 3, 4, 5-more
   persons      2, 4, more
   lug_boot     small, med, big
   safety       low, med, high
"""

import numpy as np
import itertools

#get dataset
def readDataset():
	path=raw_input("Enter the Path:")
	titles=['1','2','3','4','5','6','7']
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
		newVector=[titles[j]+'_'+vector[j] for j in range(len(titles))]
		lexicon.update(newVector)
		dictData[i]=newVector
		i+=1
	f.close()
	lexicon=sorted(lexicon)
	return dictData,titles,lexicon


def tf(word,vector):
	return vector.count(word)

#function for building feature vector
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

#get count for each candidate
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

#create dictionary from list
def getDictItemset(f2itemset):
	dictItemset={}
	for candidate in f2itemset:
		dictkey=','.join(candidate[0])
		dictItemset[dictkey]=candidate[1]
	return dictItemset

#function for getting Frequent 2 itemsets
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

#function for identifying maximal freq Itemsets
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

#function for identifying freq Closed Itemsets
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

#function for getting candidate itemset
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

#f(k-1)*f(1) method
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

#get candidate itemset
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

#f(k-1)*f(k-1) method
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
	#itemsetList=fk_1k_1(vectList,minsupport)


if __name__=="__main__":
	main()



#output

"""
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\car.csv
Enter Minimum support %: 1
minimum support count :  17
columns decoded as ColumnIndex_value
column names are as below
['1_high', '1_low', '1_med', '1_vhigh', '2_high', '2_low', '2_med', '2_vhigh', '3_2', '3_3', '3_4', '3_5more', '4_2', '4_4', '4_more', '5_big', '5_med', '5_small', '6_high', '6_low', '6_med', '7_acc', '7_good', '7_unacc', '7_vgood']
*********************
Using method Fk-1*F1
*********************
######################################################################
for F1 itemset
######################################################################
number of generated candidate itemsets:  25
number of Frequent itemsets:  25
######################################################################
######################################################################
for F2 itemset
######################################################################
number of generated candidate itemsets:  252
number of Frequent itemsets:  248
**********************************************************************
Maximal Frequent itemset count :  0
maximal Frequent itemsets are :  []
**********************************************************************
Closed Frequent itemset count :  3
Closed Frequent itemsets are :  ['4_2', '6_low', '7_vgood']
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  1341
number of Frequent itemsets:  1198
**********************************************************************
Maximal Frequent itemset count :  4
maximal Frequent itemsets are :  [['3_3', '7_good'], ['3_4', '7_good'], ['3_5more', '7_good'], ['5_med', '7_good']]
**********************************************************************
Closed Frequent itemset count :  52
Closed Frequent itemsets are :  ['3_2,6_low', '3_4,4_2', '5_med,6_low', '3_3,4_2', '1_med,7_good', '4_2,6_low', '3_4,6_low', '2_med,7_good', '1_low,6_low', '1_med,7_vgood', '1_med,4_2', '3_5more,6_low', '3_2,4_2', '2_med,6_low', '1_vhigh,2_vhigh', '5_big,7_vgood', '4_2,6_high', '1_vhigh,4_2', '4_2,5_small', '3_5more,4_2', '4_2,6_med', '1_low,4_2', '3_4,7_vgood', '4_more,7_vgood', '4_2,5_med', '5_big,6_low', '2_low,4_2', '1_low,7_vgood', '1_high,4_2', '1_high,6_low', '2_med,7_vgood', '1_med,6_low', '2_vhigh,6_low', '5_small,7_good', '4_4,6_low', '5_small,6_low', '2_vhigh,4_2', '2_low,6_low', '3_3,6_low', '1_high,2_vhigh', '1_vhigh,6_low', '2_low,7_vgood', '2_med,4_2', '4_2,5_big', '3_5more,7_vgood', '4_4,7_vgood', '4_more,6_low', '2_high,4_2', '2_high,6_low', '5_big,7_good', '5_med,7_vgood', '1_vhigh,2_high']
######################################################################
######################################################################
for F 4  itemset
######################################################################
number of generated candidate itemsets:  4073
number of Frequent itemsets:  820
**********************************************************************
Maximal Frequent itemset count :  176
maximal Frequent itemsets are :  [['1_high', '2_high', '3_4'], ['1_high', '2_high', '3_5more'], ['1_high', '2_low', '3_4'], ['1_high', '2_low', '3_5more'], ['1_high', '2_med', '3_4'], ['1_high', '2_med', '3_5more'], ['1_high', '3_2', '7_acc'], ['1_low', '2_high', '3_2'], ['1_low', '2_high', '3_3'], ['1_low', '2_high', '3_4'], ['1_low', '2_high', '3_5more'], ['1_low', '2_high', '4_more'], ['1_low', '2_high', '6_high'], ['1_low', '2_low', '3_2'], ['1_low', '2_low', '3_3'], ['1_low', '2_low', '3_4'], ['1_low', '2_low', '3_5more'], ['1_low', '2_low', '4_4'], ['1_low', '2_low', '4_more'], ['1_low', '2_low', '6_high'], ['1_low', '2_low', '6_med'], ['1_low', '2_low', '7_good'], ['1_low', '2_med', '3_2'], ['1_low', '2_med', '3_3'], ['1_low', '2_med', '3_4'], ['1_low', '2_med', '3_5more'], ['1_low', '2_med', '4_4'], ['1_low', '2_med', '4_more'], ['1_low', '2_med', '6_high'], ['1_low', '2_med', '6_med'], ['1_low', '2_med', '7_good'], ['1_low', '2_vhigh', '3_4'], ['1_low', '2_vhigh', '3_5more'], ['1_low', '3_2', '4_4'], ['1_low', '3_2', '6_high'], ['1_low', '3_2', '7_acc'], ['1_low', '3_3', '4_4'], ['1_low', '3_3', '4_more'], ['1_low', '3_3', '6_high'], ['1_low', '3_3', '6_med'], ['1_low', '3_3', '7_acc'], ['1_low', '3_4', '4_4'], ['1_low', '3_4', '4_more'], ['1_low', '3_4', '6_high'], ['1_low', '3_4', '6_med'], ['1_low', '3_4', '7_acc'], ['1_low', '3_5more', '4_4'], ['1_low', '3_5more', '4_more'], ['1_low', '3_5more', '6_high'], ['1_low', '3_5more', '6_med'], ['1_low', '3_5more', '7_acc'], ['1_low', '4_4', '5_big'], ['1_low', '4_4', '7_good'], ['1_low', '4_more', '5_big'], ['1_low', '4_more', '5_med'], ['1_low', '4_more', '7_good'], ['1_low', '5_big', '6_med'], ['1_low', '5_big', '7_acc'], ['1_low', '5_med', '6_high'], ['1_low', '6_high', '7_good'], ['1_low', '6_med', '7_good'], ['1_med', '2_high', '3_4'], ['1_med', '2_high', '3_5more'], ['1_med', '2_low', '3_2'], ['1_med', '2_low', '3_3'], ['1_med', '2_low', '3_4'], ['1_med', '2_low', '3_5more'], ['1_med', '2_low', '4_4'], ['1_med', '2_low', '4_more'], ['1_med', '2_low', '6_high'], ['1_med', '2_low', '6_med'], ['1_med', '2_low', '7_good'], ['1_med', '2_med', '3_2'], ['1_med', '2_med', '3_3'], ['1_med', '2_med', '3_4'], ['1_med', '2_med', '3_5more'], ['1_med', '2_med', '4_more'], ['1_med', '2_med', '6_high'], ['1_med', '2_vhigh', '3_4'], ['1_med', '2_vhigh', '3_5more'], ['1_med', '3_2', '4_4'], ['1_med', '3_2', '6_high'], ['1_med', '3_2', '7_acc'], ['1_med', '3_3', '4_4'], ['1_med', '3_3', '4_more'], ['1_med', '3_3', '6_high'], ['1_med', '3_3', '7_acc'], ['1_med', '3_4', '4_4'], ['1_med', '3_4', '4_more'], ['1_med', '3_4', '6_high'], ['1_med', '3_4', '6_med'], ['1_med', '3_4', '7_acc'], ['1_med', '3_5more', '4_4'], ['1_med', '3_5more', '4_more'], ['1_med', '3_5more', '6_high'], ['1_med', '3_5more', '6_med'], ['1_med', '3_5more', '7_acc'], ['1_med', '5_big', '6_high'], ['1_med', '6_high', '7_vgood'], ['1_vhigh', '2_low', '3_4'], ['1_vhigh', '2_low', '3_5more'], ['1_vhigh', '2_med', '3_4'], ['1_vhigh', '2_med', '3_5more'], ['1_vhigh', '3_3', '7_acc'], ['1_vhigh', '3_4', '7_acc'], ['1_vhigh', '3_5more', '7_acc'], ['1_vhigh', '5_big', '7_acc'], ['1_vhigh', '5_med', '7_acc'], ['1_vhigh', '6_med', '7_acc'], ['2_high', '3_2', '7_acc'], ['2_high', '3_3', '7_acc'], ['2_high', '3_4', '7_acc'], ['2_high', '3_5more', '7_acc'], ['2_low', '3_2', '4_4'], ['2_low', '3_2', '6_high'], ['2_low', '3_2', '7_acc'], ['2_low', '3_3', '4_4'], ['2_low', '3_3', '4_more'], ['2_low', '3_3', '6_high'], ['2_low', '3_3', '7_acc'], ['2_low', '3_4', '4_4'], ['2_low', '3_4', '4_more'], ['2_low', '3_4', '6_high'], ['2_low', '3_4', '6_med'], ['2_low', '3_4', '7_acc'], ['2_low', '3_5more', '4_4'], ['2_low', '3_5more', '4_more'], ['2_low', '3_5more', '6_high'], ['2_low', '3_5more', '6_med'], ['2_low', '3_5more', '7_acc'], ['2_low', '4_4', '5_big'], ['2_low', '4_4', '7_good'], ['2_low', '4_more', '5_big'], ['2_low', '4_more', '7_good'], ['2_low', '5_big', '6_high'], ['2_low', '5_big', '6_med'], ['2_low', '5_big', '7_acc'], ['2_low', '5_med', '6_high'], ['2_low', '5_med', '7_acc'], ['2_low', '5_small', '7_acc'], ['2_low', '6_high', '7_good'], ['2_low', '6_high', '7_vgood'], ['2_low', '6_med', '7_good'], ['2_med', '3_2', '4_4'], ['2_med', '3_2', '6_high'], ['2_med', '3_2', '7_acc'], ['2_med', '3_3', '4_4'], ['2_med', '3_3', '4_more'], ['2_med', '3_3', '6_high'], ['2_med', '3_3', '7_acc'], ['2_med', '3_4', '4_4'], ['2_med', '3_4', '4_more'], ['2_med', '3_4', '6_high'], ['2_med', '3_4', '6_med'], ['2_med', '3_4', '7_acc'], ['2_med', '3_5more', '4_4'], ['2_med', '3_5more', '4_more'], ['2_med', '3_5more', '6_high'], ['2_med', '3_5more', '6_med'], ['2_med', '3_5more', '7_acc'], ['2_med', '5_big', '6_high'], ['2_med', '6_high', '7_vgood'], ['2_vhigh', '3_3', '7_acc'], ['2_vhigh', '3_4', '7_acc'], ['2_vhigh', '3_5more', '7_acc'], ['2_vhigh', '5_big', '7_acc'], ['2_vhigh', '5_med', '7_acc'], ['2_vhigh', '6_med', '7_acc'], ['3_4', '6_high', '7_vgood'], ['3_5more', '6_high', '7_vgood'], ['4_4', '6_high', '7_good'], ['4_4', '6_med', '7_good'], ['4_more', '6_med', '7_good'], ['5_big', '6_med', '7_good'], ['5_med', '6_high', '7_vgood'], ['5_small', '6_high', '7_good']]
**********************************************************************
Closed Frequent itemset count :  286
Closed Frequent itemsets are :  ['1_med,2_low,4_2', '1_med,3_4,6_low', '1_vhigh,2_med,6_low', '2_med,4_2,5_small', '1_high,2_vhigh,5_small', '1_high,2_vhigh,5_med', '1_high,3_4,4_2', '2_low,5_med,6_low', '1_vhigh,2_vhigh,3_5more', '3_3,5_big,6_low', '3_4,4_4,6_low', '2_low,4_2,6_med', '1_low,5_med,6_low', '2_med,4_4,6_low', '1_low,2_high,4_2', '3_5more,4_2,5_small', '1_vhigh,2_vhigh,4_more', '1_high,5_big,6_low', '1_low,3_5more,4_2', '1_vhigh,5_small,6_med', '2_high,4_2,5_med', '2_low,4_2,5_big', '1_low,3_3,4_2', '2_high,3_2,6_low', '3_4,5_med,6_low', '1_low,2_med,4_2', '2_high,3_5more,4_2', '1_med,3_3,6_low', '2_med,5_big,6_low', '1_high,4_2,6_high', '4_more,5_big,7_vgood', '1_vhigh,2_vhigh,6_high', '1_vhigh,2_high,5_big', '1_vhigh,2_high,5_small', '1_low,4_more,6_low', '1_high,2_vhigh,6_low', '1_high,2_high,6_low', '1_vhigh,3_2,4_2', '1_high,3_3,4_2', '1_vhigh,4_4,6_low', '1_low,2_high,6_low', '1_high,3_3,6_low', '2_low,4_more,6_low', '1_med,2_low,6_low', '1_low,3_5more,6_low', '1_vhigh,2_low,4_2', '1_vhigh,4_2,5_big', '1_vhigh,2_vhigh,5_small', '4_2,5_small,6_high', '1_high,3_5more,6_low', '1_low,3_4,4_2', '1_vhigh,2_high,6_low', '2_high,4_2,6_med', '2_low,3_3,4_2', '1_high,2_high,4_2', '2_med,3_5more,4_2', '1_low,3_2,6_low', '1_high,4_2,6_med', '1_high,5_med,6_low', '3_5more,4_2,6_low', '3_4,4_2,6_med', '3_2,4_2,5_med', '1_low,2_med,6_low', '2_vhigh,5_med,6_low', '2_low,3_5more,4_2', '2_high,5_med,6_low', '3_5more,5_big,6_low', '3_5more,4_4,6_low', '2_high,4_2,5_small', '2_low,3_3,6_low', '1_vhigh,4_2,6_med', '1_low,2_low,6_low', '3_2,5_small,6_low', '1_med,2_high,6_low', '1_vhigh,2_vhigh,4_4', '1_high,2_vhigh,6_med', '1_low,4_2,6_high', '1_vhigh,5_big,6_low', '1_vhigh,2_vhigh,4_2', '1_low,5_big,7_vgood', '2_low,3_4,6_low', '1_low,4_more,7_vgood', '1_high,2_vhigh,5_big', '2_low,4_2,6_high', '2_low,3_4,4_2', '1_high,2_vhigh,4_2', '1_high,5_small,6_low', '1_vhigh,3_4,4_2', '1_med,3_2,4_2', '1_med,5_med,6_low', '1_low,4_2,5_big', '2_vhigh,4_2,6_low', '2_low,3_5more,6_low', '1_med,3_3,4_2', '2_high,4_4,6_low', '3_2,4_more,6_low', '2_vhigh,3_5more,6_low', '3_4,4_2,5_big', '3_5more,4_2,6_med', '2_vhigh,4_2,5_big', '1_vhigh,2_vhigh,3_2', '1_vhigh,3_3,4_2', '1_med,5_big,6_low', '3_5more,4_2,5_med', '4_2,5_big,6_high', '2_vhigh,3_3,4_2', '2_med,3_4,6_low', '1_med,3_2,6_low', '2_high,3_5more,6_low', '1_vhigh,2_med,4_2', '1_med,2_med,4_2', '3_2,4_2,6_high', '2_med,5_small,6_low', '1_med,4_more,6_low', '1_vhigh,2_low,6_low', '1_med,3_5more,4_2', '2_high,4_2,5_big', '2_med,3_5more,6_low', '2_high,3_4,4_2', '2_high,3_4,6_low', '1_vhigh,2_vhigh,5_big', '2_vhigh,5_small,6_low', '1_high,4_4,6_low', '3_5more,4_2,6_high', '2_low,3_2,4_2', '2_med,3_4,4_2', '4_2,5_small,6_med', '1_high,4_2,5_med', '1_high,4_more,6_low', '1_high,3_4,6_low', '1_vhigh,2_high,4_more', '1_vhigh,5_med,6_low', '2_med,3_2,6_low', '3_3,5_med,6_low', '1_vhigh,4_2,6_high', '1_vhigh,2_vhigh,3_4', '1_low,5_big,6_low', '1_vhigh,2_vhigh,3_3', '3_5more,4_2,5_big', '2_vhigh,3_4,6_low', '1_med,3_4,4_2', '3_5more,5_med,6_low', '1_high,2_vhigh,3_3', '1_high,2_vhigh,3_2', '1_high,2_vhigh,3_4', '3_3,4_2,6_med', '4_more,5_med,6_low', '1_med,5_small,6_low', '1_med,3_5more,6_low', '1_low,5_small,6_low', '1_high,4_2,6_low', '4_4,5_big,7_vgood', '2_med,5_med,6_low', '1_low,4_4,6_low', '1_med,4_4,6_low', '3_3,4_2,5_small', '1_med,2_high,4_2', '2_high,4_2,6_high', '2_high,4_more,6_low', '2_vhigh,5_small,6_med', '1_vhigh,5_small,6_low', '1_med,4_2,5_med', '1_low,3_3,6_low', '2_med,3_2,4_2', '1_low,4_4,7_vgood', '1_high,2_low,4_2', '2_med,4_2,5_big', '2_high,3_2,4_2', '1_low,3_2,4_2', '1_med,4_2,5_big', '2_med,4_2,5_med', '1_med,2_vhigh,4_2', '2_vhigh,4_2,6_high', '1_vhigh,3_5more,6_low', '2_vhigh,4_2,5_med', '1_vhigh,2_high,3_4', '4_2,5_med,6_med', '4_2,5_med,6_high', '2_vhigh,3_3,6_low', '3_2,4_2,5_small', '3_5more,5_small,6_low', '1_low,4_2,5_med', '3_4,4_2,6_high', '2_low,3_2,6_low', '1_vhigh,4_2,5_med', '1_vhigh,2_high,4_2', '1_vhigh,3_3,6_low', '1_vhigh,2_high,4_4', '3_3,4_2,6_low', '1_vhigh,2_vhigh,5_med', '3_4,5_small,6_low', '2_low,4_2,6_low', '1_low,2_low,4_2', '1_high,2_med,4_2', '4_more,5_big,6_low', '2_high,4_2,6_low', '2_high,5_big,6_low', '1_low,2_vhigh,4_2', '1_high,5_small,6_med', '1_vhigh,2_vhigh,6_low', '1_low,4_2,6_low', '3_4,4_2,6_low', '2_low,5_small,6_low', '2_vhigh,4_4,6_low', '3_2,5_med,6_low', '1_vhigh,3_5more,4_2', '1_high,3_2,6_low', '3_3,5_small,6_low', '1_high,2_vhigh,4_more', '3_3,4_2,5_big', '3_2,4_more,5_small', '3_3,4_4,6_low', '2_high,3_3,4_2', '1_high,4_2,5_big', '1_low,2_vhigh,6_low', '1_vhigh,4_2,5_small', '1_vhigh,3_2,6_low', '1_high,3_2,4_2', '1_low,3_4,6_low', '1_low,4_2,6_med', '1_med,2_med,6_low', '1_med,2_vhigh,6_low', '1_vhigh,3_4,6_low', '3_2,4_2,5_big', '1_vhigh,2_high,3_2', '2_vhigh,3_5more,4_2', '1_high,2_vhigh,3_5more', '1_med,4_2,6_high', '3_4,4_more,6_low', '2_med,4_2,6_low', '1_vhigh,2_high,6_med', '1_high,3_5more,4_2', '2_med,3_3,6_low', '1_med,4_2,6_med', '3_3,4_2,5_med', '2_vhigh,4_2,6_med', '2_vhigh,3_2,6_low', '1_vhigh,2_high,3_5more', '3_3,4_2,6_high', '3_4,4_2,5_small', '2_low,5_big,6_low', '3_2,4_4,6_low', '4_2,5_big,6_med', '2_vhigh,5_big,6_low', '1_vhigh,4_2,6_low', '3_4,4_2,5_med', '2_low,4_4,6_low', '2_med,4_more,6_low', '2_low,4_2,5_small', '4_2,5_med,6_low', '3_4,5_big,6_low', '3_2,4_2,6_med', '2_high,5_small,6_low', '2_vhigh,3_4,4_2', '1_high,5_small,7_acc', '1_high,4_2,5_small', '1_vhigh,2_high,6_high', '3_5more,4_more,6_low', '2_high,3_3,6_low', '1_med,4_2,6_low', '2_med,4_2,6_high', '1_vhigh,2_high,5_med', '2_med,4_2,6_med', '3_2,4_2,6_low', '3_3,4_more,6_low', '1_vhigh,2_high,3_3', '2_vhigh,4_2,5_small', '4_2,5_small,6_low', '1_high,2_med,6_low', '4_4,5_med,6_low', '1_high,2_vhigh,6_high', '3_2,5_big,6_low', '2_vhigh,3_2,4_2', '2_med,3_3,4_2', '4_4,5_small,6_low', '1_vhigh,2_vhigh,6_med', '4_4,5_big,6_low', '1_low,4_2,5_small', '2_vhigh,4_more,6_low', '1_high,2_vhigh,4_4', '1_med,4_2,5_small', '2_low,4_2,5_med', '1_vhigh,4_more,6_low', '4_more,5_small,6_low', '4_2,5_big,6_low', '1_high,2_low,6_low']
######################################################################

C:\Users\samee\OneDrive\Assignments\DM assignent 4\Question3>python apriori_c_car.py
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\car.csv
Enter Minimum support %: 5
minimum support count :  86
columns decoded as ColumnIndex_value
column names are as below
['1_high', '1_low', '1_med', '1_vhigh', '2_high', '2_low', '2_med', '2_vhigh', '3_2', '3_3', '3_4', '3_5more', '4_2', '4_4', '4_more', '5_big', '5_med', '5_small', '6_high', '6_low', '6_med', '7_acc', '7_good', '7_unacc', '7_vgood']
*********************
Using method Fk-1*F1
*********************
######################################################################
for F1 itemset
######################################################################
number of generated candidate itemsets:  25
number of Frequent itemsets:  23
######################################################################
######################################################################
for F2 itemset
######################################################################
number of generated candidate itemsets:  223
number of Frequent itemsets:  220
**********************************************************************
Maximal Frequent itemset count :  0
maximal Frequent itemsets are :  []
**********************************************************************
Closed Frequent itemset count :  2
Closed Frequent itemsets are :  ['4_2', '6_low']
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  1175
number of Frequent itemsets:  106
**********************************************************************
Maximal Frequent itemset count :  89
maximal Frequent itemsets are :  [['1_high', '2_high'], ['1_high', '2_low'], ['1_high', '2_med'], ['1_high', '3_3'], ['1_high', '3_4'], ['1_high', '3_5more'], ['1_high', '6_high'], ['1_high', '7_acc'], ['1_low', '2_high'], ['1_low', '2_low'], ['1_low', '2_med'], ['1_low', '2_vhigh'], ['1_low', '3_2'], ['1_low', '3_3'], ['1_low', '3_4'], ['1_low', '3_5more'], ['1_low', '4_4'], ['1_low', '4_more'], ['1_low', '5_big'], ['1_low', '5_med'], ['1_low', '6_high'], ['1_low', '6_med'], ['1_low', '7_acc'], ['1_med', '2_high'], ['1_med', '2_low'], ['1_med', '2_med'], ['1_med', '2_vhigh'], ['1_med', '3_2'], ['1_med', '3_3'], ['1_med', '3_4'], ['1_med', '3_5more'], ['1_med', '4_4'], ['1_med', '4_more'], ['1_med', '5_big'], ['1_med', '5_med'], ['1_med', '6_high'], ['1_med', '6_med'], ['1_med', '7_acc'], ['1_vhigh', '2_low'], ['1_vhigh', '2_med'], ['2_high', '3_2'], ['2_high', '3_3'], ['2_high', '3_4'], ['2_high', '3_5more'], ['2_high', '4_4'], ['2_high', '4_more'], ['2_high', '6_high'], ['2_high', '7_acc'], ['2_low', '3_2'], ['2_low', '3_3'], ['2_low', '3_4'], ['2_low', '3_5more'], ['2_low', '4_4'], ['2_low', '4_more'], ['2_low', '5_big'], ['2_low', '5_med'], ['2_low', '6_high'], ['2_low', '6_med'], ['2_low', '7_acc'], ['2_med', '3_2'], ['2_med', '3_3'], ['2_med', '3_4'], ['2_med', '3_5more'], ['2_med', '4_4'], ['2_med', '4_more'], ['2_med', '5_big'], ['2_med', '5_med'], ['2_med', '6_high'], ['2_med', '6_med'], ['2_med', '7_acc'], ['3_2', '4_4'], ['3_2', '6_high'], ['3_3', '4_4'], ['3_3', '4_more'], ['3_3', '6_high'], ['3_3', '7_acc'], ['3_4', '4_4'], ['3_4', '4_more'], ['3_4', '6_high'], ['3_4', '6_med'], ['3_4', '7_acc'], ['3_5more', '4_4'], ['3_5more', '4_more'], ['3_5more', '6_high'], ['3_5more', '6_med'], ['3_5more', '7_acc'], ['5_big', '7_acc'], ['5_med', '7_acc'], ['5_small', '7_acc']]
**********************************************************************
Closed Frequent itemset count :  38
Closed Frequent itemsets are :  ['3_2,6_low', '3_4,4_2', '5_med,6_low', '3_3,4_2', '4_2,6_low', '3_4,6_low', '1_low,6_low', '1_med,4_2', '3_5more,6_low', '3_2,4_2', '2_med,6_low', '1_vhigh,2_vhigh', '4_2,6_high', '1_vhigh,4_2', '4_2,5_small', '3_5more,4_2', '4_2,6_med', '1_low,4_2', '4_2,5_med', '2_high,6_low', '2_low,4_2', '1_high,4_2', '1_high,6_low', '1_med,6_low', '2_vhigh,6_low', '5_small,6_low', '2_vhigh,4_2', '2_low,6_low', '3_3,6_low', '1_high,2_vhigh', '1_vhigh,6_low', '2_med,4_2', '4_2,5_big', '5_big,6_low', '4_more,6_low', '2_high,4_2', '4_4,6_low', '1_vhigh,2_high']
######################################################################


C:\Users\samee\OneDrive\Assignments\DM assignent 4\Question3>python apriori_c_car.py
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\car.csv
Enter Minimum support %: 7
minimum support count :  120
columns decoded as ColumnIndex_value
column names are as below
['1_high', '1_low', '1_med', '1_vhigh', '2_high', '2_low', '2_med', '2_vhigh', '3_2', '3_3', '3_4', '3_5more', '4_2', '4_4', '4_more', '5_big', '5_med', '5_small', '6_high', '6_low', '6_med', '7_acc', '7_good', '7_unacc', '7_vgood']
*********************
Using method Fk-1*F1
*********************
######################################################################
for F1 itemset
######################################################################
number of generated candidate itemsets:  25
number of Frequent itemsets:  23
######################################################################
######################################################################
for F2 itemset
######################################################################
number of generated candidate itemsets:  223
number of Frequent itemsets:  162
**********************************************************************
Maximal Frequent itemset count :  0
maximal Frequent itemsets are :  []
**********************************************************************
Closed Frequent itemset count :  2
Closed Frequent itemsets are :  ['4_2', '6_low']
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  586
number of Frequent itemsets:  41
**********************************************************************
Maximal Frequent itemset count :  100
maximal Frequent itemsets are :  [['1_high', '4_4'], ['1_high', '4_more'], ['1_high', '5_big'], ['1_high', '5_med'], ['1_high', '6_high'], ['1_high', '6_med'], ['1_low', '4_4'], ['1_low', '4_more'], ['1_low', '5_big'], ['1_low', '5_med'], ['1_low', '5_small'], ['1_low', '6_high'], ['1_low', '6_med'], ['1_med', '4_4'], ['1_med', '4_more'], ['1_med', '5_big'], ['1_med', '5_med'], ['1_med', '5_small'], ['1_med', '6_high'], ['1_med', '6_med'], ['1_vhigh', '4_4'], ['1_vhigh', '4_more'], ['1_vhigh', '5_big'], ['1_vhigh', '5_med'], ['1_vhigh', '6_high'], ['1_vhigh', '6_med'], ['2_high', '4_4'], ['2_high', '4_more'], ['2_high', '5_big'], ['2_high', '5_med'], ['2_high', '5_small'], ['2_high', '6_high'], ['2_high', '6_med'], ['2_low', '4_4'], ['2_low', '4_more'], ['2_low', '5_big'], ['2_low', '5_med'], ['2_low', '5_small'], ['2_low', '6_high'], ['2_low', '6_med'], ['2_med', '4_4'], ['2_med', '4_more'], ['2_med', '5_big'], ['2_med', '5_med'], ['2_med', '5_small'], ['2_med', '6_high'], ['2_med', '6_med'], ['2_vhigh', '4_4'], ['2_vhigh', '4_more'], ['2_vhigh', '5_big'], ['2_vhigh', '5_med'], ['2_vhigh', '6_high'], ['2_vhigh', '6_med'], ['3_2', '4_4'], ['3_2', '4_more'], ['3_2', '5_big'], ['3_2', '5_med'], ['3_2', '6_high'], ['3_2', '6_med'], ['3_3', '4_4'], ['3_3', '4_more'], ['3_3', '5_big'], ['3_3', '5_med'], ['3_3', '5_small'], ['3_3', '6_high'], ['3_3', '6_med'], ['3_4', '4_4'], ['3_4', '4_more'], ['3_4', '5_big'], ['3_4', '5_med'], ['3_4', '5_small'], ['3_4', '6_high'], ['3_4', '6_med'], ['3_5more', '4_4'], ['3_5more', '4_more'], ['3_5more', '5_big'], ['3_5more', '5_med'], ['3_5more', '5_small'], ['3_5more', '6_high'], ['3_5more', '6_med'], ['4_4', '5_big'], ['4_4', '5_med'], ['4_4', '5_small'], ['4_4', '6_high'], ['4_4', '6_med'], ['4_4', '7_acc'], ['4_more', '5_big'], ['4_more', '5_med'], ['4_more', '6_high'], ['4_more', '6_med'], ['4_more', '7_acc'], ['5_big', '6_high'], ['5_big', '6_med'], ['5_big', '7_acc'], ['5_med', '6_high'], ['5_med', '6_med'], ['5_med', '7_acc'], ['5_small', '6_high'], ['6_high', '7_acc'], ['6_med', '7_acc']]
**********************************************************************
Closed Frequent itemset count :  35
Closed Frequent itemsets are :  ['3_2,6_low', '3_3,4_2', '3_4,4_2', '5_med,6_low', '4_2,6_low', '3_4,6_low', '1_low,6_low', '1_med,4_2', '3_5more,6_low', '3_2,4_2', '2_med,6_low', '4_2,6_high', '1_vhigh,4_2', '4_2,5_small', '3_5more,4_2', '4_2,6_med', '1_low,4_2', '4_2,5_med', '2_high,6_low', '2_low,4_2', '1_high,4_2', '1_high,6_low', '1_med,6_low', '2_vhigh,6_low', '5_small,6_low', '2_vhigh,4_2', '2_low,6_low', '3_3,6_low', '1_vhigh,6_low', '2_med,4_2', '4_2,5_big', '5_big,6_low', '4_more,6_low', '2_high,4_2', '4_4,6_low']
######################################################################
"""