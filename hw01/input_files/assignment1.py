import numpy as np
import os
import random
import math
from treelib import Node, Tree
dataset=[]  # would be used as testing set ,its a list
trainset=[]
incset=[]   # add increment in each time
properties={}

def testTree():
	pass
def train(increment):
	test=[['f', 'f', 'e', 't', 'n', 'f', 'c', 'b', 'n', 't', 'b', 's', 's', 'p', 'g', 'p', 'w', 'o', 'p', 'k', 'v', 'd', 'e'], 
			['f', 'y', 'g', 'f', 'f', 'f', 'c', 'b', 'h', 'e', 'b', 'k', 'k', 'n', 'p', 'p', 'w', 'o', 'l', 'h', 'y', 'd', 'p'], 
			['f', 'f', 'y', 'f', 'f', 'f', 'c', 'b', 'g', 'e', 'b', 'k', 'k', 'n', 'b', 'p', 'w','o', 'l', 'h', 'v', 'g', 'p'], 
			['f', 'y', 'n', 't', 'n', 'f', 'c', 'b', 'w', 't', 'b', 's', 's', 'g', 'g', 'p', 'w', 'o', 'p', 'k', 'y', 'd', 'e'], 
			['f', 'y','e', 't', 'n', 'f', 'c', 'b', 'u', 't', 'b', 's', 's', 'p', 'p', 'p', 'w', 'o','p', 'k', 'v', 'd', 'e']]
	# print(GetSubExamples(test,1))
	incset=trainset[0:200]
	attributes=[]
	for i in range(22):
		attributes.append(i)
	# print(getImportant(incset,attributes))
	trained=Tree()
	trainedTree=DecitionTreeLearning(incset,attributes,[])
	trainedTree.show(line_type="ascii-em")

def DecitionTreeLearning(examples,attributes,pexamples):
	if len(examples)==0:
		return  getPlurality(pexamples)
	elif isSameClassification(examples):
		SameClassTree = Tree()
		if examples[0][len(examples[0])-1]=='e':
			SameClassTree.create_node("Edible")
		else:
			SameClassTree.create_node("Poison")
		return SameClassTree
	elif len(attributes)==0:
		return getPlurality(examples)
	else:
		node=getImportant(examples,attributes)
		tree = Tree()
		tree.create_node(str(getPropertyName(node)), node)
		# tree.show()
		subExamples=getSubExamples(examples,node)
		attributes.remove(node)
		for key, value in subExamples.items():
			nodeIdentifier=getPropertyName(node)+"_"+key
			tree.create_node(key,nodeIdentifier,node)
			subtree=DecitionTreeLearning(value,attributes,examples)
			tree.paste(nodeIdentifier,subtree)
		return tree


def getPropertyName(target):
	index=0
	for key in properties:
		if index==target:
			return key
		index+=1

def getSubExamples(targetSet,attr):          #works correct
	global properties
	dic={}
	index=0
	attrVal=[]
	for key in properties:
		if index==attr:
			attrVal=properties[key]
		index+=1
	for i in attrVal:
		dic[i]=list()
	for item in targetSet:
		result=item[attr]
		if result in dic:
			dic[result].append(item)
	return dic
		
def getPlurality(targetSet):
	countP=0
	countE=0
	for item in targetSet:
		if item[len(item)-1]=='p':
			countP+=1
		else:
			countE+=1
	tree = Tree()
	if countE>countP:
		tree.create_node("Edible")
		return tree
	elif countE<countP:
		tree.create_node("Poison")
		return tree
	else :
		if random.randint(0,1)==1:
			tree.create_node("Edible")
			return tree
		else:
			tree.create_node("Poison")
			return tree

def isSameClassification(targetSet):
	countP=0
	for item in targetSet:
		if item[len(item)-1]=='p':
			countP+=1
	if (countP==len(targetSet))|(countP==0):
		return True

def getImportant(incSet,attributes):
	importants=[]
	for i in attributes:
		importants.append(getGain(incSet,i))
	maxVal=max(importants)
	# print("importants")
	# print(importants)
	# print("importants.index(maxVal)")
	# print(importants.index(maxVal))
	return attributes[importants.index(maxVal)]

def getGain(targetSet,attr):
	dic={}
	infEntropy=getEntropy(targetSet,len(targetSet[0])-1)
	# print("infEntropy: "+str(infEntropy))
	infGain=infEntropy
	for item in targetSet:
		result=item[attr]
		if result in dic:
			dic[result].append(item)
		else:
			dic[result]=list()
			dic[result].append(item)
	# print(dic)
	for value in dic.values():
		entropy=getEntropy(value,22)
		infGain-=entropy*len(value)/len(targetSet)
	return infGain
	# print("infGain: "+str(infGain))

def getEntropy(targetSet,attr):             #works correct
	dic={}
	for item in targetSet:
		result=item[attr]
		if result in dic:
			dic[result]+=1
		else:
			dic.setdefault(result,1)
	# consider the 0 condition (Seems do not need concern)
	# print(dic)
	entropy=0.0
	for value in dic.values():
		val=-(value/len(targetSet))*math.log(value/len(targetSet),2)
		entropy=entropy+val
	return entropy

def getData(trainSize,increment):
	global dataset,trainset,properties
	# trainSize=250
	# increment=10
	print("Training Set Size: "+str(trainSize))
	print("Increment of Training Set: "+str(increment))
	# with open(r'''C:\Users\Administrator\Desktop\hw01\hw01\input_files\mushroom_data.txt''', "r") as f:  # fow windows to get dataset
	with open(r'''/Users/leijinhuan/Documents/Code/hw01/input_files/mushroom_data.txt''', "r") as f:       # fow Mac
		# data=f.readlines()
		for line in f.readlines():
			line=line.strip()
			dataset.append(list(map(str,line.split(" "))))
		# print(dataset[0]) 
		# print(len(data))  #5644
	# with open(r'''C:\Users\Administrator\Desktop\hw01\hw01\input_files\properties.txt''', "r") as l:      #properties
	with open(r'''/Users/leijinhuan/Documents/Code/hw01/input_files/properties.txt''', "r") as l:
		for line in l.readlines():
			line=line.strip()
			content=line.split(":")
			content[1]=content[1].strip()
			properties[content[0]]=list(map(str,content[1].split(" ")))
	# print(properties)
	for i in range(trainSize):
		index = random.randint(0,len(dataset)-1)
		trainset.append(dataset[index])
		dataset.pop(index)
	# print("trainset:"+str(len(trainset)))
	# print(trainset[0:5])
	train(increment)

def GetInputs():
	trainSize=0
	increment=0
	trainSize=int(input("Please Enter Training Set Size(250 ≤ S ≤ 1000):"))
	if (trainSize>1000)|(trainSize<250):
		print("Please Enter the Effective Value!")
		GetInputs()
	while(True):
		increment=int(input("Please Enter the Increment of Training Set(10,25,50):"))
		if (increment==10)|(increment==25)|(increment==50):
			break
		print("Please Enter the Specified Value!")
	getData(trainSize,increment)

if __name__=="__main__":
	getData(250,10)