# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:13:58 2018

@author: Borja042
"""

import numpy as np
import pywt
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt

#Hay que hacer dos veces la decomposicion. No se parece a la funcion implementada porque el filtro es diferente.
#En la segunda parte se coge LL1 y se vuelve a dividir y así te vas quedando con la primera LL que es la que mas info contiene.
# Se pueden hacer hasta 7 divisiones que se llega a dos pixels. 

original = mpimg.imread('lena512.bmp')
total = []
medias = []
diferencias = []

matriz = np.array([[1,2,1,2,1,2], [0,1,0,1,0,1], [2,3,2,3,2,3],[3,4,3,4,3,4],[1,2,1,2,1,2], [0,1,0,1,0,1]])
mitad_medias = []
#mitad_medias2 = np.array()
mitad_diferencias = []

n_fila = 0
for line in original:
#    print(i)
    n_fila += 1
    sizeList = line.size
    listaPares=np.split(line, sizeList/2)
    #print(listaPares)
    
    for mitad in listaPares:
        #print(np.average(mitad))
        mitad_medias += [float(np.average(mitad))]
        mitad_diferencias += [(mitad[0]-mitad[1])/2]
        
        
#      
mitad_medias = np.array(mitad_medias).reshape(original.shape[0],int(original.shape[1]/2))
mitad_diferencias = np.array(mitad_diferencias).reshape(original.shape[0],int(original.shape[1]/2))
matriz_final = np.concatenate((mitad_medias, mitad_diferencias), axis = 1)

imgplot = plt.imshow(matriz_final,cmap='gray')

matriz_final2 = matriz_final.transpose()
#imgplot = plt.imshow(matriz_final2,cmap='gray')





total = []
medias = []
diferencias = []

mitad_medias = []
#mitad_medias2 = np.array()
mitad_diferencias = []


for line in matriz_final2:
#    print(i)
    
    sizeList = line.size
    listaPares=np.split(line, sizeList/2)
    #print(listaPares)
    
    for mitad in listaPares:
        #print(np.average(mitad))
        mitad_medias += [float(np.average(mitad))]
        mitad_diferencias += [(mitad[0]-mitad[1])/2]


cuarto_medias1, cuarto_medias2 = np.split(np.array(mitad_medias), 2)
cuarto_diferencias1, cuarto_diferencias2 = np.split(np.array(mitad_diferencias), 2)

cuarto_medias1 = np.array(cuarto_medias1).reshape(int(original.shape[0]/2),int(original.shape[1]/2))
cuarto_medias2 = np.array(cuarto_medias2).reshape(int(original.shape[0]/2),int(original.shape[1]/2))
cuarto_diferencias1 = np.array(cuarto_diferencias1).reshape(int(original.shape[0]/2),int(original.shape[1]/2))
cuarto_diferencias2 = np.array(cuarto_diferencias2).reshape(int(original.shape[0]/2),int(original.shape[1]/2))

#final1 = np.concatenate((cuarto_medias1, cuarto_medias2), axis = 1)
#final2 = np.concatenate((cuarto_diferencias1, cuarto_diferencias2), axis = 1)
#final_final = np.concatenate((final1, final2), axis = 0).transpose()
#
#imgplot = plt.imshow(final_final,cmap='gray')


#################################################
#################################################


total = []
medias = []
diferencias = []

mitad_medias = []
#mitad_medias2 = np.array()
mitad_diferencias = []


for line in cuarto_medias1:
#    print(i)

    sizeList = line.size
    listaPares=np.split(line, sizeList/2)
    #print(listaPares)
    
    for mitad in listaPares:
        #print(np.average(mitad))
        mitad_medias += [float(np.average(mitad))]
        mitad_diferencias += [(mitad[0]-mitad[1])/2]
        
        
#      
mitad_medias = np.array(mitad_medias).reshape(int(cuarto_medias1.shape[0]),int(cuarto_medias1.shape[1]/2))
mitad_diferencias = np.array(mitad_diferencias).reshape(int(cuarto_medias1.shape[0]),int(cuarto_medias1.shape[1]/2))
matriz_final = np.concatenate((mitad_medias, mitad_diferencias), axis = 1)

imgplot = plt.imshow(matriz_final,cmap='gray')

matriz_final2 = matriz_final.transpose()
imgplot = plt.imshow(matriz_final2,cmap='gray')

total = []
medias = []
diferencias = []

mitad_medias = []
#mitad_medias2 = np.array()
mitad_diferencias = []


##############


for line in matriz_final2:
#    print(i)
    
    sizeList = line.size
    listaPares=np.split(line, sizeList/2)
    #print(listaPares)
    
    for mitad in listaPares:
        #print(np.average(mitad))
        mitad_medias += [float(np.average(mitad))]
        mitad_diferencias += [(mitad[0]-mitad[1])/2]


cm1, cm2 = np.split(np.array(mitad_medias), 2)
cd1, cd2 = np.split(np.array(mitad_diferencias), 2)

cm1 = np.array(cm1).reshape(int(matriz_final2.shape[0]/2),int(matriz_final2.shape[1]/2))
cm2 = np.array(cm2).reshape(int(matriz_final2.shape[0]/2),int(matriz_final2.shape[1]/2))
cd1 = np.array(cd1).reshape(int(matriz_final2.shape[0]/2),int(matriz_final2.shape[1]/2))
cd2 = np.array(cd2).reshape(int(matriz_final2.shape[0]/2),int(matriz_final2.shape[1]/2))

ff1 = np.concatenate((cm1, cm2), axis = 1)
ff2 = np.concatenate((cd1, cd2), axis = 1)
cm1_final = np.concatenate((ff1, ff2), axis = 0).transpose()

final1 = np.concatenate((cm1_final, cuarto_medias2), axis = 1)
final2 = np.concatenate((cuarto_diferencias1, cuarto_diferencias2), axis = 1)
final_final = np.concatenate((final1, final2), axis = 0).transpose()

imgplot = plt.imshow(final_final,cmap='gray')
