from numpy import *
import math
from numpy.linalg import *
import json
#from pylab import *
import random
import os
import matplotlib.pyplot

#The following parameters need to be changed for each implementation:
#  "minmaxManual.txt" x 2;
#  to generate test traj for 5th Ring, uncomment the start-end segment with 4th Ring minmaxManual;
#  to generate traj with more timestamps, change test_timestamps;

#  do not change g=50, which means 50 x 50 grids;
#  test==1 means we only need one test trajectory, do not change this;
#  "index=len(data)-4" determines which traj is the test traj;
#  end;

rootdir = os.path.join("..", os.getcwd())
resultdir = os.path.join(rootdir, "results")
#data_file="geolife_dataset_024.txt"
#Mat_file="TransMat_024.txt"
data_file = os.path.join(resultdir, "geolife_dataset_027.txt")
Mat_file = os.path.join(resultdir, "TransMat_027.txt")
#reading the trajectories, removing testSize of them for test data
def getData(**args):
    data = dict()    
    if(len(args.items())==0):
        args['real'] = True
        args['test'] = 100
	
    testSize = args['test']
    #print("test size: "+str(testSize))
    #raw_input(str(testSize))
    
    if(args['real']!=True):    
        mapX = 1000.0
        mapY = 1000.0
        map0X = 0
        map0Y = 0
        data = syndata.createSynData(1000,1000)
    else:
        with open(os.path.join(rootdir, "params", "minmaxManual_4thRing.txt")) as f: 
            map0X = float(f.readline().strip())
            map0Y = float(f.readline().strip())
            mapX = float(f.readline().strip())
            mapY = float(f.readline().strip())
        f.close()
        data = list()    
        with open(data_file) as f: 
            data = json.loads(f.read())
        
        print("Number of Trajectories: " + str(len(data)))
        
        #testIndices = random.sample(range(len(data)),testSize)
        testData = list()   
        for i in range(0,testSize):
            #index = random.randrange(0,len(data))
            index=len(data)-4
            testData.append(data[index])    
            del data[index]
        
        print("Number of Test Trajectories: "+ str(len(testData)))
    return list((data,([map0X,map0Y,mapX,mapY]),testData))

def createTransMatrix1st(g,data,mapInfo):
    map0X = mapInfo[0]
    map0Y = mapInfo[1]
    mapX = mapInfo[2]
    mapY = mapInfo[3]
    mapDimensionX = mapX-map0X
    mapDimensionY = mapY-map0Y
    matrix = zeros((g*g,g*g))
    nodeCounts = zeros((g*g))    
    cellWidth = mapDimensionX/g
    cellHeight = mapDimensionY/g     
    for litem in data:
        item = asarray(litem)
        shape = item.shape
        #plot(item[:,1],item[:,2],'r.')
        for i in range(0,shape[0]-1):
            currentX = int((item[i,1]-map0X)/cellWidth)    
            currentY = int((item[i,2]-map0Y)/cellHeight)
            current = currentX+currentY*g            
            if(i==0):
                nodeCounts[current] = nodeCounts[current] + 1
            nextX = int((item[i+1,1]-map0X-0.0001)/cellWidth)    
            nextY = int((item[i+1,2]-map0Y)/cellHeight)
            next = nextX+nextY*g
			#if you want to find the probability of staying in the same cell remove the if condition  (It depends how far in the future you need the transition probability          
            #if(current!=next):
            
                
            nodeCounts[current] = nodeCounts[current] + 1
            matrix[current,next] = matrix[current,next] + 1
    #xlim(map0X,mapX)
    #ylim(map0Y,mapY)        
    #show()
    for i in range(0,g*g):
        for j in range(0,g*g):
            if(nodeCounts[i]!=0):    
                matrix[i,j] = matrix[i,j]/nodeCounts[i]    


    return matrix  


def genTestTraj(g,timestamps,test_data,mapInfo):
    map0X = mapInfo[0]
    map0Y = mapInfo[1]
    mapX = mapInfo[2]
    mapY = mapInfo[3]
    mapDimensionX = mapX-map0X
    mapDimensionY = mapY-map0Y
    traj_matrix = zeros((timestamps,1))
    traj_coord=zeros((2,timestamps))
    #nodeCounts = zeros((g*g))    
    cellWidth = mapDimensionX/g
    cellHeight = mapDimensionY/g     
    for litem in test_data:
        item = asarray(litem)
        shape = item.shape
        #print(size(item))
        #plot(item[1:timestamps,1],item[1:timestamps,2],'r.')
        #plot(item[230,1],item[230,2],'b*')
        #plot(item[240,1],item[240,2],'bs')
        #plot(item[250,1],item[250,2],'b+')
        #plot(item[300,1],item[300,2],'bo')
        #plot(item[330,1],item[330,2],'b^')
        
        for i in range(0,shape[0]-1):
            if(i>timestamps-1):
                break
            currentX = int((item[i,1]-map0X)/cellWidth)    
            currentY = int((item[i,2]-map0Y)/cellHeight)
            #plot((item[i,1]-map0X)/cellWidth,(item[i,2]-map0Y)/cellHeight,'r.')
            
            ##if(i>=231 and i<=330):
            ##    plot((item[i,1]-map0X)/cellWidth,(item[i,2]-map0Y)/cellHeight,'b+')
            traj_coord[0,i]=currentX+0.5;
            traj_coord[1,i]=currentY+0.5;
            current = currentX+currentY*g            
            
            traj_matrix[i,0] = int(current)
            
        #plot((item[0,1]-map0X)/cellWidth,(item[0,2]-map0Y)/cellHeight,'b*')
        #plot((item[timestamps-1,1]-map0X)/cellWidth,(item[timestamps-1,2]-map0Y)/cellHeight,'bs')
        
        #plot(traj_coord[0,:],traj_coord[1,:],'ro')
        #plot(traj_coord[0,:],traj_coord[1,:],'r-')
        #show()
            
    print(size(traj_matrix))
    return traj_matrix  


#g is the map grid granularity 
g = 50
test_timestamps=500;
data = getData(real = True, test = 1)
trainData = data[0]
mapInfo = data[1]
testData = data[2]
M = createTransMatrix1st(g,trainData,mapInfo)
savetxt(Mat_file,M)
##start: to generate traj for 5thRing;
#with open("minmaxManual.txt") as f: 
#    map0X = float(f.readline().strip())
#    map0Y = float(f.readline().strip())
#    mapX = float(f.readline().strip())
#    mapY = float(f.readline().strip())
#f.close()
#mapInfo=[map0X,map0Y,mapX,mapY]
##end: to generate traj for 5thRing;

#test_traj=genTestTraj(g,test_timestamps,testData,mapInfo)
#savetxt('testTraj.txt',test_traj)
print(M)
