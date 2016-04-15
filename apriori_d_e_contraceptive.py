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
import operator
import pandas as pd

no=0

#get Dataset
def readDataset():
	path=raw_input("Enter the Path:")
	#path="C:\\Users\\samee\\OneDrive\\Assignments\\DM assignent 4\\nursery.csv"
	# noOfCols=int(raw_input("Enter number of columns:"))
	# i=0
	# titles=[]
	titleNames=['wifeAge','wifeEdu','husbandEdu','childrenNo','wifeReligion','wifeWorking','husbandOcu','SOL','mediaExposure','contMethod']
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
	return dictData,titles,lexicon,titleNames

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
	f1Itemset={key: value for key,value in f1Itemset.items() if value>0} #remove itemset if their value is 0
	f1Itemset={key: value for key,value in f1Itemset.items() if value>minsupport}
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
	f2dict={key: value for key,value in f2dict.items() if value>minsupport}
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
		reqItemsetDict={key: value for key,value in reqItemsetDict.items() if value>minsupport}
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
		reqItemsetDict={key: value for key,value in reqItemsetDict.items() if value>minsupport}
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

def duplicateRule(rules,item):
	for rule in rules:
		if rules[0]==item:
			return True
	return False

def calculateConfidence(lhs,rhs,itemsetListAll):
	lhsRule=','.join(lhs)
	x=sorted(lhs+rhs)
	lenx=len(x)
	x=','.join(x)
	dictlhs=itemsetListAll[len(lhs)]
	dictall=itemsetListAll[lenx]
	ruleConfidence=float(dictall[x])/dictlhs[lhsRule]
	ruleSupport=float(dictall[x])/no
	return ruleConfidence,ruleSupport

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

#if lhs present in nonConfidence list prune the rule
def confidencePrune(nonConfidenceList,lhsList):
	for candidate in nonConfidenceList:
		if checkSuperSet(candidate,lhsList):
			return True
	return False

def generateIRules(itemsetList,k,allRules,itemsetListAll,confidenceThreshold,confidenceDict,supportDict):
	freqItemset=[]
	rules=[]
	nonConfidenceList=[]
	for candidate in itemsetList:
		freqItemset.append(candidate.split(','))
	for itemset in freqItemset:
		for i in range(1,len(itemset)):
			nItemList=[list(x) for x in itertools.combinations(itemset,i)]
			for items in nItemList:
				tempList=sorted(list(set(itemset)-set(items)))
				if not duplicateRule(allRules,tempList):
					confidence,support=calculateConfidence(tempList,items,itemsetListAll)
					rules.append([tempList,items,confidence]) #contains all generated rules
					if(confidence>confidenceThreshold): #append to required dictionary
						if(not confidencePrune(nonConfidenceList,tempList)):
							tempList=','.join(tempList)
							items=','.join(items)
							temp=tempList+'|'+items
							confidenceDict[temp]=confidence
							supportDict[temp]=support
					else:
						nonConfidenceList.append(tempList) #contains items not satisfying min threshold for pruning
	return rules,confidenceDict,nonConfidenceList


def genRules(itemsetList,titleNames):
	print "Starting rule generation..."
	print "Enter 3 values of confidence..."
	for i in range(0,3):
		confidence=float(raw_input("Enter "+str(i+1)+"/"+"3"+" Confidence Threshold : "))
		allRules=[]
		confidenceDict={}
		supportDict={}
		for i in range (len(itemsetList)-1,1,-1):
			if(len(itemsetList[i]))>0:
				rules,confidenceDict,nonConfidenceList=generateIRules(itemsetList[i],i,allRules,itemsetList,confidence,confidenceDict,supportDict)
				for rule in rules:
					allRules.append(rule)
		# print confidenceDict
		# print nonConfidenceList
		print "#"*70
		print "Total rules generated by bruteforce",len(allRules)
		print "Total rules generated after confidence pruning",len(confidenceDict)
		ip=raw_input("Do you want to see all the rules?(1->yes): ")
		if ip=='1':
			printRules(confidenceDict,supportDict,len(confidenceDict),titleNames)
		print "Printing top 10 rules by confidence"
		printRules(confidenceDict,supportDict,10,titleNames)
		print "#"*70
	return confidenceDict

#input: ColumnIndex_columnValue
#output: columnName_columnValue
def getTitle(candidate,titleNames):
	candidate=candidate.split('_')
	index=int(candidate[0])-1
	return titleNames[index]+"."+candidate[1]

def parseRule(rule,titleNames):
	parsedLhs=""
	parsedRhs=""
	lhsrhs=rule.split('|')
	lhs=lhsrhs[0]
	rhs=lhsrhs[1]
	lhs=lhs.split(',')
	rhs=rhs.split(',')
	if len(lhs)==1:
		parsedLhs=parsedLhs+getTitle(lhs[0],titleNames)
	else:
		for rule in lhs:
			parsedLhs=parsedLhs+getTitle(rule,titleNames)+', '
		parsedLhs=parsedLhs[:-1]
	if len(rhs)==1:
		parsedRhs=parsedRhs+getTitle(rhs[0],titleNames)
	else:
		for rule in rhs:
			parsedRhs=parsedRhs+getTitle(rule,titleNames)+', '
		parsedRhs=parsedRhs[:-1]
	return parsedLhs+"->"+parsedRhs

def printRules(rulesDict,supportDict,count,titleNames):
	pd.set_option('max_colwidth', 1200)
	keys=[]
	confidence=[]
	support=[]
	for key,value in rulesDict.items():
		rule=parseRule(key,titleNames)
		keys.append(rule)
		confidence.append(value)
		support.append(supportDict[key])
	df=pd.DataFrame({'Rules':keys,'Confidence':confidence,'Support':support})
	df.sort_values(['Confidence','Support'],ascending=False,inplace=True)
	cols=df.columns.tolist()
	cols=[cols[1]] +[cols[0]] +[cols[2]]
	df=df[cols]
	if count==len(rulesDict):
		print df.to_string(index=False)
	else:
		print df.head(n=count).to_string(index=False)
	
	return

def main():
	data,titles,lexicon,titleNames=readDataset()
	global no
	no=len(data)
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
	rulesDict=genRules(itemsetList,titleNames)
	#printRules(rulesDict,10,titleNames)
	#itemsetList=fk_1k_1(vectList,minsupport)


if __name__=="__main__":
	main()

#output1
"""
Enter the Path:contraceptive.csv
Enter Minimum support %: 20
minimum support count :  294
Using method Fk-1*F1
Starting rule generation...
Enter 3 values of confidence...
Enter 1/3 Confidence Threshold %: 0.3
######################################################################
Total rules generated by bruteforce 1426
Total rules generated after confidence pruning 1049
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                                  Rules  Confidence   Support
                                     wifeEdu.4, SOL.4,->mediaExposure.0    0.994819  0.260692
                       wifeEdu.4, husbandEdu.4, SOL.4,->mediaExposure.0    0.994652  0.252546
                       wifeEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.993631  0.211813
         wifeEdu.4, husbandEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.993528  0.208418
               wifeEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.990881  0.221317
                            wifeEdu.4, wifeReligion.1,->mediaExposure.0    0.990783  0.291921
             wifeEdu.4, wifeReligion.1, wifeWorking.1,->mediaExposure.0    0.990506  0.212492
 wifeEdu.4, husbandEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.990415  0.210455
                              wifeEdu.4, husbandOcu.1,->mediaExposure.0    0.990354  0.209097
                             wifeEdu.4, wifeWorking.1,->mediaExposure.0    0.990172  0.273591
######################################################################
Enter 2/3 Confidence Threshold %: 0.5
######################################################################
Total rules generated by bruteforce 1426
Total rules generated after confidence pruning 700
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                                  Rules  Confidence   Support
                                     wifeEdu.4, SOL.4,->mediaExposure.0    0.994819  0.260692
                       wifeEdu.4, husbandEdu.4, SOL.4,->mediaExposure.0    0.994652  0.252546
                       wifeEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.993631  0.211813
         wifeEdu.4, husbandEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.993528  0.208418
               wifeEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.990881  0.221317
                            wifeEdu.4, wifeReligion.1,->mediaExposure.0    0.990783  0.291921
             wifeEdu.4, wifeReligion.1, wifeWorking.1,->mediaExposure.0    0.990506  0.212492
 wifeEdu.4, husbandEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.990415  0.210455
                              wifeEdu.4, husbandOcu.1,->mediaExposure.0    0.990354  0.209097
                             wifeEdu.4, wifeWorking.1,->mediaExposure.0    0.990172  0.273591
######################################################################
Enter 3/3 Confidence Threshold %: 0.8
######################################################################
Total rules generated by bruteforce 1426
Total rules generated after confidence pruning 161
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                                  Rules  Confidence   Support
                                     wifeEdu.4, SOL.4,->mediaExposure.0    0.994819  0.260692
                       wifeEdu.4, husbandEdu.4, SOL.4,->mediaExposure.0    0.994652  0.252546
                       wifeEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.993631  0.211813
         wifeEdu.4, husbandEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.993528  0.208418
               wifeEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.990881  0.221317
             wifeEdu.4, wifeReligion.1, wifeWorking.1,->mediaExposure.0    0.990506  0.212492
 wifeEdu.4, husbandEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.990415  0.210455
                              wifeEdu.4, husbandOcu.1,->mediaExposure.0    0.990354  0.209097
                wifeEdu.4, husbandEdu.4, husbandOcu.1,->mediaExposure.0    0.990164  0.205024
                                             wifeEdu.4->mediaExposure.0    0.989601  0.387644
######################################################################

Output2
Enter the Path:contraceptive.csv
Enter Minimum support %: 25
minimum support count :  368
Using method Fk-1*F1
Starting rule generation...
Enter 3 values of confidence...
Enter 1/3 Confidence Threshold %: 0.3
######################################################################
Total rules generated by bruteforce 706
Total rules generated after confidence pruning 644
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                     Rules  Confidence   Support
                        wifeEdu.4, SOL.4,->mediaExposure.0    0.994819  0.260692
          wifeEdu.4, husbandEdu.4, SOL.4,->mediaExposure.0    0.994652  0.252546
               wifeEdu.4, wifeReligion.1,->mediaExposure.0    0.990783  0.291921
                wifeEdu.4, wifeWorking.1,->mediaExposure.0    0.990172  0.273591
 wifeEdu.4, husbandEdu.4, wifeReligion.1,->mediaExposure.0    0.990123  0.272234
                                wifeEdu.4->mediaExposure.0    0.989601  0.387644
                 wifeEdu.4, childrenNo.1,->mediaExposure.0    0.989583  0.322471
  wifeEdu.4, husbandEdu.4, wifeWorking.1,->mediaExposure.0    0.989583  0.257977
   wifeEdu.4, husbandEdu.4, childrenNo.1,->mediaExposure.0    0.989059  0.306857
                 wifeEdu.4, husbandEdu.4,->mediaExposure.0    0.988971  0.365241
######################################################################
Enter 2/3 Confidence Threshold %: 0.5
######################################################################
Total rules generated by bruteforce 706
Total rules generated after confidence pruning 369
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                     Rules  Confidence   Support
                        wifeEdu.4, SOL.4,->mediaExposure.0    0.994819  0.260692
          wifeEdu.4, husbandEdu.4, SOL.4,->mediaExposure.0    0.994652  0.252546
               wifeEdu.4, wifeReligion.1,->mediaExposure.0    0.990783  0.291921
                wifeEdu.4, wifeWorking.1,->mediaExposure.0    0.990172  0.273591
 wifeEdu.4, husbandEdu.4, wifeReligion.1,->mediaExposure.0    0.990123  0.272234
                                wifeEdu.4->mediaExposure.0    0.989601  0.387644
                 wifeEdu.4, childrenNo.1,->mediaExposure.0    0.989583  0.322471
  wifeEdu.4, husbandEdu.4, wifeWorking.1,->mediaExposure.0    0.989583  0.257977
   wifeEdu.4, husbandEdu.4, childrenNo.1,->mediaExposure.0    0.989059  0.306857
                 wifeEdu.4, husbandEdu.4,->mediaExposure.0    0.988971  0.365241
######################################################################
Enter 3/3 Confidence Threshold %: 0.8
######################################################################
Total rules generated by bruteforce 706
Total rules generated after confidence pruning 93
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                     Rules  Confidence   Support
                        wifeEdu.4, SOL.4,->mediaExposure.0    0.994819  0.260692
          wifeEdu.4, husbandEdu.4, SOL.4,->mediaExposure.0    0.994652  0.252546
               wifeEdu.4, wifeReligion.1,->mediaExposure.0    0.990783  0.291921
                wifeEdu.4, wifeWorking.1,->mediaExposure.0    0.990172  0.273591
 wifeEdu.4, husbandEdu.4, wifeReligion.1,->mediaExposure.0    0.990123  0.272234
                                wifeEdu.4->mediaExposure.0    0.989601  0.387644
                 wifeEdu.4, childrenNo.1,->mediaExposure.0    0.989583  0.322471
  wifeEdu.4, husbandEdu.4, wifeWorking.1,->mediaExposure.0    0.989583  0.257977
   wifeEdu.4, husbandEdu.4, childrenNo.1,->mediaExposure.0    0.989059  0.306857
       husbandEdu.4, childrenNo.1, SOL.4,->mediaExposure.0    0.980861  0.278344
######################################################################

Output3
Enter the Path:contraceptive.csv
Enter Minimum support %: 30
minimum support count :  441
Using method Fk-1*F1
Starting rule generation...
Enter 3 values of confidence...
Enter 1/3 Confidence Threshold : 0.3
######################################################################
Total rules generated by bruteforce 282
Total rules generated after confidence pruning 282
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                        Rules  Confidence   Support
                                   wifeEdu.4->mediaExposure.0    0.989601  0.387644
                    wifeEdu.4, childrenNo.1,->mediaExposure.0    0.989583  0.322471
      wifeEdu.4, husbandEdu.4, childrenNo.1,->mediaExposure.0    0.989059  0.306857
                    wifeEdu.4, husbandEdu.4,->mediaExposure.0    0.988971  0.365241
                        husbandEdu.4, SOL.4,->mediaExposure.0    0.980769  0.346232
  husbandEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.977316  0.350984
                       wifeWorking.1, SOL.4,->mediaExposure.0    0.975510  0.324508
                 husbandEdu.4, childrenNo.1,->mediaExposure.0    0.975342  0.483367
 husbandEdu.4, childrenNo.1, wifeReligion.1,->mediaExposure.0    0.973822  0.378819
                                       SOL.4->mediaExposure.0    0.973684  0.452138
######################################################################
Enter 2/3 Confidence Threshold %: 0.5
######################################################################
Total rules generated by bruteforce 282
Total rules generated after confidence pruning 169
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                        Rules  Confidence   Support
                                   wifeEdu.4->mediaExposure.0    0.989601  0.387644
                    wifeEdu.4, childrenNo.1,->mediaExposure.0    0.989583  0.322471
      wifeEdu.4, husbandEdu.4, childrenNo.1,->mediaExposure.0    0.989059  0.306857
                    wifeEdu.4, husbandEdu.4,->mediaExposure.0    0.988971  0.365241
                        husbandEdu.4, SOL.4,->mediaExposure.0    0.980769  0.346232
  husbandEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.977316  0.350984
                       wifeWorking.1, SOL.4,->mediaExposure.0    0.975510  0.324508
                 husbandEdu.4, childrenNo.1,->mediaExposure.0    0.975342  0.483367
 husbandEdu.4, childrenNo.1, wifeReligion.1,->mediaExposure.0    0.973822  0.378819
                                       SOL.4->mediaExposure.0    0.973684  0.452138
######################################################################
Enter 3/3 Confidence Threshold %: 0.8
######################################################################
Total rules generated by bruteforce 282
Total rules generated after confidence pruning 58
Do you want to see all the rules?(1->yes): 2
Printing top 10 rules by confidence
                                                        Rules  Confidence   Support
                                   wifeEdu.4->mediaExposure.0    0.989601  0.387644
                    wifeEdu.4, childrenNo.1,->mediaExposure.0    0.989583  0.322471
      wifeEdu.4, husbandEdu.4, childrenNo.1,->mediaExposure.0    0.989059  0.306857
                    wifeEdu.4, husbandEdu.4,->mediaExposure.0    0.988971  0.365241
                        husbandEdu.4, SOL.4,->mediaExposure.0    0.980769  0.346232
  husbandEdu.4, childrenNo.1, wifeWorking.1,->mediaExposure.0    0.977316  0.350984
                       wifeWorking.1, SOL.4,->mediaExposure.0    0.975510  0.324508
 husbandEdu.4, childrenNo.1, wifeReligion.1,->mediaExposure.0    0.973822  0.378819
                        childrenNo.1, SOL.4,->mediaExposure.0    0.973180  0.344874
                husbandEdu.4, wifeWorking.1,->mediaExposure.0    0.971726  0.443313
######################################################################

"""