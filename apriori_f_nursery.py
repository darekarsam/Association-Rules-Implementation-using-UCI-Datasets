# Author: Sameer Darekar
# Title: Implementing Apriori Algorithm
# Dataset Location: http://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data

"""
   Attribute Information
   parents        usual, pretentious, great_pret
   has_nurs       proper, less_proper, improper, critical, very_crit
   form           complete, completed, incomplete, foster
   children       1, 2, 3, more
   housing        convenient, less_conv, critical
   finance        convenient, inconv
   social         non-prob, slightly_prob, problematic
   health         recommended, priority, not_recom
   class	  recommend,priority, not_recom, very_recom,spec_prior
"""

import numpy as np
import itertools
import operator
import pandas as pd

no=0

def readDataset():
	path=raw_input("Enter the Path:")
	
	titleNames=['parents','has_nurs','form','children','housing','finance','social','health','class']
	titles=['1','2','3','4','5','6','7','8','9']
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

def duplicateRule(rules,item):
	for rule in rules:
		if rules[0]==item:
			return True
	return False

def calculateLift(lhs,rhs,itemsetListAll):
	lhsRule=','.join(lhs)
	x=sorted(lhs+rhs)
	lenx=len(x)
	x=','.join(x)
	rhsRule=','.join(rhs)
	dictrhs=itemsetListAll[len(rhs)]
	dictlhs=itemsetListAll[len(lhs)]
	dictall=itemsetListAll[lenx]
	ruleConfidence=float(dictall[x])/dictlhs[lhsRule]
	supRhs=float(dictrhs[rhsRule])/no
	lift=float(ruleConfidence)/supRhs
	ruleSupport=float(dictall[x])/no
	return lift,ruleSupport,ruleConfidence

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

# #if lhs present in nonConfidence list prune the rule
# def liftPrune(nonConfidenceList,lhsList):
# 	for candidate in nonConfidenceList:
# 		if checkSuperSet(candidate,lhsList):
# 			return True
# 	return False

def generateIRules(itemsetList,k,allRules,itemsetListAll,liftDict,supportDict,confidenceDict):
	freqItemset=[]
	rules=[]
	for candidate in itemsetList:
		freqItemset.append(candidate.split(','))
	for itemset in freqItemset:
		for i in range(1,len(itemset)):
			nItemList=[list(x) for x in itertools.combinations(itemset,i)]
			for items in nItemList:
				tempList=sorted(list(set(itemset)-set(items)))
				if not duplicateRule(allRules,tempList):
					lift,support,confidence=calculateLift(tempList,items,itemsetListAll)
					rules.append([tempList,items,lift]) #contains all generated rules
					tempList=','.join(tempList)
					items=','.join(items)
					temp=tempList+'|'+items
					liftDict[temp]=lift
					supportDict[temp]=support
					confidenceDict[temp]=confidence
	return rules,liftDict,supportDict,confidenceDict

def genRules(itemsetList,titleNames):
	print "Starting rule generation..."
	allRules=[]
	liftDict={}
	supportDict={}
	confidenceDict={}
	for i in range (len(itemsetList)-1,1,-1):
		if(len(itemsetList[i]))>0:
			rules,liftDict,supportDict,confidenceDict=generateIRules(itemsetList[i],i,allRules,itemsetList,liftDict,supportDict,confidenceDict)
			for rule in rules:
				allRules.append(rule)
	# print liftDict
	# print nonConfidenceList
	print "#"*70
	print "Total rules generated: ",len(allRules)
	print "Top 10 rules generated by lift:"
	printRules(liftDict,supportDict,confidenceDict,10,titleNames)
	print "#"*70
	return liftDict

def getTitle(candidate,titleNames):
	candidate=candidate.split('_')
	index=int(candidate[0])-1
	if len(candidate)>2:
		return titleNames[index]+"."+candidate[1]+"_"+candidate[2]
	else:
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
		parsedLhs=parsedLhs[:-2]
	if len(rhs)==1:
		parsedRhs=parsedRhs+getTitle(rhs[0],titleNames)
	else:
		for rule in rhs:
			parsedRhs=parsedRhs+getTitle(rule,titleNames)+', '
		parsedRhs=parsedRhs[:-2]
	return parsedLhs+"->"+parsedRhs

def printRules(rulesDict,supportDict,confidenceDict,count,titleNames):
	pd.set_option('max_colwidth', 1200)
	keys=[]
	lift=[]
	confidence=[]
	support=[]
	for key,value in rulesDict.items():
		rule=parseRule(key,titleNames)
		keys.append(rule)
		lift.append(float(value))
		confidence.append(confidenceDict[key])
		support.append(supportDict[key])
	df=pd.DataFrame({'Rules':keys,'Lift':lift, 'Confidence':confidence,'Support':support})
	df.sort_values(['Lift','Confidence','Support'],ascending=False,inplace=True)
	cols=df.columns.tolist()
	cols=[cols[2]]+[cols[1]] +[cols[0]] +[cols[3]]
	df=df[cols]
	if count==len(rulesDict):
		print df.to_string(index=False)
	else:
		print df.head(n=count).to_string(index=False)
	return

def main():
	global no
	data,titles,lexicon,titleNames=readDataset()
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
	#itemsetList=fk_1k_1(vectList,minsupport)
	# dictl=itemsetList[4]
	# print dictl["4_2,5_small,6_high,7_unacc"]
	# dictl=itemsetList[3]
	# print dictl["4_2,5_small,6_high"]

if __name__=="__main__":
	main()

#Output
"""
C:\Users\samee\OneDrive\Assignments\DM assignent 4>python apriori_f_nursery.py
Enter the Path:nursery.csv
Enter Minimum support %: 2
minimum support count :  259
Using method Fk-1*F1
Starting rule generation...
######################################################################
Total rules generated:  20182
Top 10 rules generated by lift:
                                                                       Rules      Lift  Confidence   Support
        parents.usual, class.spec_prior->has_nurs.very_crit, health.priority  5.639842    0.375989  0.021991
        has_nurs.very_crit, health.priority->parents.usual, class.spec_prior  5.639842    0.329861  0.021991
   has_nurs.very_crit, health.priority->housing.convenient, class.spec_prior  3.978137    0.322917  0.021528
   housing.convenient, class.spec_prior->has_nurs.very_crit, health.priority  3.978137    0.265209  0.021528
       has_nurs.very_crit, health.priority->social.nonprob, class.spec_prior  3.562500    0.329861  0.021991
 has_nurs.very_crit, health.priority->social.slightly_prob, class.spec_prior  3.562500    0.329861  0.021991
       social.nonprob, class.spec_prior->has_nurs.very_crit, health.priority  3.562500    0.237500  0.021991
 social.slightly_prob, class.spec_prior->has_nurs.very_crit, health.priority  3.562500    0.237500  0.021991
         parents.usual, class.spec_prior->has_nurs.very_crit, finance.inconv  3.509235    0.350923  0.020525
         has_nurs.very_crit, finance.inconv->parents.usual, class.spec_prior  3.509235    0.205247  0.020525
######################################################################

C:\Users\samee\OneDrive\Assignments\DM assignent 4>python apriori_f_nursery.py
Enter the Path:nursery.csv
Enter Minimum support %: 5
minimum support count :  648
Using method Fk-1*F1
Starting rule generation...
######################################################################
Total rules generated:  2314
Top 10 rules generated by lift:
                                                 Rules      Lift  Confidence   Support
 has_nurs.very_crit, health.priority->class.spec_prior  3.171365    0.989583  0.065972
 class.spec_prior->has_nurs.very_crit, health.priority  3.171365    0.211424  0.065972
                     class.not_recom->health.not_recom  3.000000    1.000000  0.333333
                     health.not_recom->class.not_recom  3.000000    1.000000  0.333333
     finance.inconv, health.not_recom->class.not_recom  3.000000    1.000000  0.166667
 finance.convenient, health.not_recom->class.not_recom  3.000000    1.000000  0.166667
     finance.inconv, class.not_recom->health.not_recom  3.000000    1.000000  0.166667
 finance.convenient, class.not_recom->health.not_recom  3.000000    1.000000  0.166667
  housing.less_conv, health.not_recom->class.not_recom  3.000000    1.000000  0.111111
   housing.critical, health.not_recom->class.not_recom  3.000000    1.000000  0.111111
######################################################################

C:\Users\samee\OneDrive\Assignments\DM assignent 4>python apriori_f_nursery.py
Enter the Path:nursery.csv
Enter Minimum support %: 8
minimum support count :  1036
Using method Fk-1*F1
Starting rule generation...
######################################################################
Total rules generated:  664
Top 10 rules generated by lift:
                                                   Rules  Lift  Confidence   Support
                       class.not_recom->health.not_recom     3           1  0.333333
                       health.not_recom->class.not_recom     3           1  0.333333
   finance.convenient, health.not_recom->class.not_recom     3           1  0.166667
       finance.inconv, class.not_recom->health.not_recom     3           1  0.166667
   finance.convenient, class.not_recom->health.not_recom     3           1  0.166667
       finance.inconv, health.not_recom->class.not_recom     3           1  0.166667
    housing.less_conv, health.not_recom->class.not_recom     3           1  0.111111
 social.slightly_prob, class.not_recom->health.not_recom     3           1  0.111111
    housing.less_conv, class.not_recom->health.not_recom     3           1  0.111111
     housing.critical, health.not_recom->class.not_recom     3           1  0.111111
######################################################################
"""