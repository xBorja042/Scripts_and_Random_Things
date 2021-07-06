# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:12:41 2018

@author: Borja042
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt.data
import pandas as pd


# Load image
img=mpimg.imread('lena512.bmp')

# Wavelet transform of image, and plot approximation and details
titles = ['Original','Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs
fig = plt.figure(figsize=(24, 6))
for i, a in enumerate([img,LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

print(len(coeffs)-1)

fig=plt.figure(figsize=(8, 8))
coeffs2 = pywt.dwt2(LL, 'haar')
LL2, (LH2, HL2, HH2) = coeffs2
fig = plt.figure(figsize=(24, 6))
for i, a in enumerate([LL,LL2, LH2, HL2, HH2]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

print(len(coeffs))


ff1 = np.concatenate((LL2,LH2), axis = 0)
ff2 = np.concatenate((HL2,HH2), axis = 0)
final = np.concatenate((ff1,ff2), axis = 1)

imgplot = plt.imshow(final,cmap='gray')

lista_aplicar = [ LH, HL, HH, LH2, HL2, HH2]
lista_imgs = ['LH1','HL1','HH1', 'LH2', 'HL2', 'HH2']

ff12 = np.concatenate((final, HL), axis = 0)
ff21 = np.concatenate((LH,HH), axis = 0)
ff2 = np.concatenate((ff12,ff21), axis = 1)


def scalar_quantizer(img, R):
    smin = np.amin(img)
    smax = np.amax(img)
    delta = float(smax - smin)/float(2**R)
    return np.floor((img-smin)/delta + 0.5)* delta + smin


def entropy(img):
#     calculate the entropy of a imgae
    img_size = img.size
    unique_pixel_list = np.unique(img)
    
    probability_list = [np.size(img[img==i]) / float(img_size) for i in unique_pixel_list]
    return np.sum([(-1) * p * np.log2(p) for p in probability_list])



entropias_originales = []
entropias_transf = []
tupla_niveles = ()

lista_imgs2 = []
lista_R = []
i = 0
lista_matrix = []
for imgOriginal in lista_aplicar:
    
    D_list = []
    print('Cuadrante -->', lista_imgs[i])
    
    for R in range(1,9):

        entropias_originales += [entropy(imgOriginal)]
        print('Valor de R -->', R)
        lista_R += [R]
        imgTransformed = scalar_quantizer(imgOriginal, R)
        if R == 5:
            lista_matrix += [imgTransformed]
        entropias_transf += [entropy(imgTransformed)]
        tupla_niveles = tupla_niveles + ("Img",  lista_imgs[i], "Valor de R", R, "Entropia" , entropy(imgTransformed))
        lista_imgs2 += [lista_imgs[i]]
        df = pd.DataFrame({'original':np.array(imgOriginal.flatten(), dtype = 'f'), 'transformed':np.array(imgTransformed.flatten(), dtype = 'f')})
        df['Img'], df['Valor de R'], df['Entropia'] = lista_imgs[i], R, entropy(imgTransformed)
        D = sum((df['original'] - df['transformed'])**2) / len(df)
        print ("Distortion with the original image =", D)
        imgplot = plt.imshow(imgTransformed,cmap='gray')
        #print("R--> ", R)
        plt.show()
        
        D_list += [D]
    i = i + 1
    plt.plot(range(1,9), D_list)
    plt.scatter(range(1,9), D_list,color='red')
    plt.title("Distortion (D) for each possible value of R")
    plt.xlabel("R value")
    plt.ylabel("D(R value)")
    plt.show()
df2 = pd.DataFrame({'img':  lista_imgs2,'valor_R':  lista_R , 'entropia_img_trasf': entropias_transf , 'entropia_img_original':entropy(imgOriginal)})
df2['comp_ratio'] = (df2['entropia_img_trasf'] / df2['entropia_img_original'])
# Los plots de la distorsion dicen que los niveles guapos son los 4-5, habría que
# hablar de esos bien.
lh1_hh1 = np.concatenate((lista_matrix[0], lista_matrix[2]), axis = 1)
#h11_hh1 = np.concatenate((lista_matrix[2], lista_matrix[3]), axis = 0)



ff11 = np.concatenate((LL2,lista_matrix[3]), axis = 0)
ff22 = np.concatenate((lista_matrix[3],lista_matrix[4]), axis = 0)
cuarto1 = np.concatenate((ff11,ff22), axis = 1)

level2_hl1 = np.concatenate((cuarto1, lista_matrix[1]), axis = 1)

final_inversa = np.concatenate((level2_hl1,lh1_hh1), axis = 0)

imgplot = plt.imshow(final_inversa,cmap='gray')

max_levels = R

img_quantized = final_inversa
# Sintesis
next_level=np.array(img_quantized[:int(len(img_quantized)/(2**(max_levels-1))),:int(len(img_quantized)/(2**(max_levels-1)))])       
for k in range(8, 0, -1):
    
    # Operamos por columnas (para ello hacemos la transp, operamos por filas y hacemos transp del resultado)
    transp = np.array(next_level).transpose()
    recomp_matrix=[]
    for line in transp:
        orig=[]
        for i in range(0, int(line.size/2)):
            mean = line[i]
            dif = line[i+int(line.size/2)]
            v1 = mean+dif
            v2 = mean-dif
            orig = orig + [v1, v2] 
        recomp_matrix = recomp_matrix + [orig]
    recomp_matrix = np.array(recomp_matrix).transpose()
        
    # Operamos por filas
    recomp_matrix2=[]
    for line in recomp_matrix:
        orig=[]
        for i in range(0, int(line.size/2)):
            mean = line[i]
            dif = line[i+int(line.size/2)]
            v1 = mean+dif
            v2 = mean-dif
            orig = orig + [v1, v2] 
        recomp_matrix2 = recomp_matrix2 + [orig]
    
    h = len(recomp_matrix2)
    w = len(recomp_matrix2[0])
    img_quantized[:h,:w] = recomp_matrix2
    next_level = np.array(img_quantized[:2*h, :2*w])
    next_level[:h, :w] = recomp_matrix2
    
reconstructed = np.array(img_quantized)
print("QUANTIZED IMAGE - RECONSTRUCTED IMAGE")
imgplot = plt.imshow(reconstructed, cmap='gray')
plt.show()