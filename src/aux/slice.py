# -*- coding: utf-8 -*-

"""
    slice a MOVEMENT file into images.
    return a number of randonly chosen images

    L. Wang 2022.7
"""
import sys
import numpy as np 

from numpy.random import choice 

arg_list = sys.argv[1:]
#arg_list = arg_list[2:]

def slice(f_name, num_select = None):
    #for f_name in arg_list:
    
    #print (f_name) 
    
    filename = f_name
    filename_out = f_name + ".shuffle" 
    
    file = open(filename,"r")
    lines = file.readlines()
    
    file.close()  
    
    #sample_dist = 13     # distance between sampled images 
    allImage = []
    singleImage = []  
    
    for line in lines:
        #encountering a new image 
        if len(line.split())>2 and line.split()[1] == 'atoms,Iteration' and len(singleImage)!=0:
            
            allImage.append(singleImage.copy())
            
            singleImage.clear()
            
        singleImage.append(line)
    
    allImage.append(singleImage.copy())

    numImage = len(allImage)          

    #print ("number of total image:",numImage) 

    shuffledImage = choice(numImage, numImage ,replace = False)  
        
    res = [] 

    for imageIdx in shuffledImage[:num_select]:
        
        res.append(allImage[imageIdx])
        """
        for line in allImage[imageIdx]:
            file.write(line)
        """

    return res 
