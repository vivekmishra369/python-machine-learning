import numpy as np
import operator
import pandas as pd



    
if __name__ == "__main__":
    print ("Welcome to K-nearest neighbor supervised learning classifier.\n\
All you will need to do is give  your dataset and your unclassified feature set and this program will classify it for you\n\
Please keep in mind that the data must be in a csv file or else you will have to modify the source code\n\
\n\
so lets begin...")

da=[]
mother_file=raw_input("please mention the name of your mother file\n")
da = pd.read_csv(mother_file)
d=np.mat(da)
dataset=d[:,0:-1]
labels=d[:,-1]        


inx=raw_input("please enter your input parameters in the form of a list ")

x=np.mat(inx)
if x.shape[1]!=dataset.shape[1]:
    print ("invalid input\n please run the program again")
    
kay=raw_input("how many nearest neighbours(k) would you like to count the majority from\n")
k=int(kay)
siz=dataset.shape[0]
bl=np.zeros([dataset.shape[0],dataset.shape[1]])
for i in range(len(bl)):
    bl[i,:]=x
difmat=bl-dataset

sqdifmat=np.square(difmat.astype(float))
root=np.sqrt(sqdifmat)
distances = root.sum(axis=1)

sortedDistIndicies = np.argsort(distances,axis=0)     
classCount={} 

for i in range(k):
    
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
print sortedClassCount[0][0]       

