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
	# path=path.replace("\\","\\\\")
	#path="C:\\Users\\samee\\OneDrive\\Assignments\\DM assignent 4\\car.csv"
	# noOfCols=int(raw_input("Enter number of columns:"))
	# i=0
	# titles=[]
	# for i in range(noOfCols):
	# 	titles.append(raw_input("Enter Column Name:"))
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

def getCount(itemset,vectors):
	f2itemset=[]
	lexicon=vectors[0].tolist()
	vectors=vectors[1:]
	for candidate in itemset:
		candCount=0
		indexVec=[lexicon.index(x) for x in candidate]
		for vector in vectors:
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
	print "number of Frequent itemsets: ",len(f2dict)
	print "#"*70
	return f2dict

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
	if len(reqItemsetDict)>0:
		print "#"*70
		print "for F",k," itemset"
		print "#"*70
		print "number of generated candidate itemsets: ",len(reqItemsetDict)
		reqItemsetDict={key: value for key,value in reqItemsetDict.items() if value>minsupport}
		print "number of Frequent itemsets: ",len(reqItemsetDict)
		print "#"*70
	return reqItemsetDict

def fk_1(vectors,minsupport):
	itemsetList=[0]
	print"Using method Fk-1*F1"
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
		print "number of Frequent itemsets: ",len(reqItemsetDict)
		print "#"*70
	return reqItemsetDict

def fk_1k_1(vectors,minsupport):
	itemsetList=[0]
	print"Using method Fk-1*Fk-1"
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
	itemsetList=fk_1(vectList,minsupport)
	fk_1k_1(vectList,minsupport)


if __name__=="__main__":
	main()

#output
"""
C:\Users\samee\OneDrive\Assignments\DM assignent 4>python apriori_a_b_car.py
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\car.csv
Enter Minimum support %: 5
minimum support count :  86
Using method Fk-1*F1
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
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  1175
number of Frequent itemsets:  106
######################################################################
Using method Fk-1*Fk-1
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
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  1173
number of Frequent itemsets:  106
######################################################################

C:\Users\samee\OneDrive\Assignments\DM assignent 4>python apriori_a_b_car.py
Enter the Path:C:\Users\samee\OneDrive\Assignments\DM assignent 4\car.csv
Enter Minimum support %: 10
minimum support count :  172
Using method Fk-1*F1
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
number of Frequent itemsets:  52
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  70
number of Frequent itemsets:  11
######################################################################
Using method Fk-1*Fk-1
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
number of Frequent itemsets:  52
######################################################################
######################################################################
for F 3  itemset
######################################################################
number of generated candidate itemsets:  70
number of Frequent itemsets:  11
######################################################################
"""