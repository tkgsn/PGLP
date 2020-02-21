#!/usr/bin/python
# -*- coding: utf-8 -*-


#The following parameters need to be changed for each implementation:
#  "minmaxManual.txt" x 1 (to "minmaxManual_4thRing.txt")
# end;




#Original Dataset attributes:
#lat,long,,,time in days,date,time



#from numpy import *
import datetime
import latlongutil as ll
import json
import os
#from pylab import *

def readnlines(f, n):
    lines = []
    for x in range(0, n):
        lines.append(f.readline())
    return lines

rootdir = os.path.join("..", os.getcwd())
resultdir = os.path.join(rootdir, "results")

def getRealData(address):
    minValueX = 10e10
    minValueY = 10e10
    maxValueX = -10e10
    maxValueY = -10e10
	#Beijing's bounds
    with open(os.path.join(rootdir, "params","minmaxManual_4thRing.txt")) as f: 
        map0X = float(f.readline().strip())
        map0Y = float(f.readline().strip())
        mapX = float(f.readline().strip())
        mapY = float(f.readline().strip())
    f.close()
    print(map0X)
    print(map0Y)
    print(mapX)
    print(mapY)
    print("distance x, distance y: ")
    print(ll.distance_on_unit_sphere(map0Y,map0X,mapY,map0X))
    print(ll.distance_on_unit_sphere(map0Y,map0X,map0Y,mapX))
    
    allTrajs = list()
    for root, dirs, files in os.walk(address, topdown=False):
        print(root)
        print(files)
        for name in files:
            if (name!="labels.txt"):
                traj = list()            
                path = os.path.join(root, name)            
                with open(path) as f:
                    uselessLines = readnlines(f,6)
                    lines = f.read().splitlines()
                f.close()
                for each in lines:
                    eachList = each.split(',')
                    if(len(eachList) < 5):
                        print(path)                        
                        print(eachList)
                    currentTime = float(eachList[4])    
                    dimensionXY = ([float(eachList[1]), float(eachList[0])])
                    if (dimensionXY[0] < minValueX):
                        minValueX = dimensionXY[0]
                    if (dimensionXY[1] < minValueY):
                        minValueY = dimensionXY[1]
                    if (dimensionXY[0] > maxValueX):
                        maxValueX = dimensionXY[0]
                    if (dimensionXY[1] > maxValueY):
                        maxValueY = dimensionXY[1]
                    #plot(dimensionXY[0],dimensionXY[1],'r.')  
                    if not(dimensionXY[0]<=map0X or dimensionXY[1]<=map0Y or dimensionXY[0]>=mapX or dimensionXY[1]>=mapY):
                        traj.append([currentTime,dimensionXY[0],dimensionXY[1]])
                    
                if(len(traj)>=3):
                    allTrajs.append(traj)    
    #show()
    print(len(allTrajs))
    with open(os.path.join(resultdir, "minmax.txt"),'w') as g: 
        g.write(str(minValueX) + os.linesep)
        g.write(str(minValueY) + os.linesep)
        g.write(str(maxValueX) + os.linesep)
        g.write(str(maxValueY) + os.linesep)
    g.close()


    #for writting the test traj;
    #traj_data=list()
    #traj_data.append(allTrajs[len(allTrajs)-4])
    #g=50
    #matrix = zeros((2000,2))
    #mapDimensionX = mapX-map0X
    #mapDimensionY = mapY-map0Y  
    #cellWidth = mapDimensionX/g
    #cellHeight = mapDimensionY/g
    #for litem in traj_data:
    #    item = asarray(litem)
    #    shape = item.shape
    #    for i in range(0,shape[0]-1):
    #        if(i>=2000):
    #            break;
    #        currentX = ((item[i,1]-map0X)/cellWidth)    
    #        currentY = ((item[i,2]-map0Y)/cellHeight)
    #        matrix[i,0]=currentX;
    #        matrix[i,1]=currentY;
    #savetxt('geolife_test_traj.txt',matrix)
                
    return(allTrajs)

#Read the data, preprocess it: extract the trajectories within 5th Rings of Beijing and store it in the file
with open(os.path.join(resultdir, "geolife_dataset_027.txt"), "w") as f: 
    f.write(json.dumps(getRealData(os.path.join(rootdir, "Geolife Trajectories 1.3", "Data", "027", "Trajectory"))))
    #f.write(json.dumps(getRealData('./test_traj')))
f.close()
 
