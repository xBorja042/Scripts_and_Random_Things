# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:10:14 2022

@author: f.gonzalez
"""

def scaler():
    from sklearn.preprocessing import StandardScaler
    print("Escala datos y deja la media a 0 y la std a 1")
    data = [[0, 1], [2, 3], [-1, 1], [-4, 0], [-1, -7], [9, 9]]
    print(f"Initial data {data}")
    scaler = StandardScaler()
    scaler.fit(data)
    sc_data = scaler.transform(data)
    print(" datos escalados ", "\n", sc_data)
    print(f"Media y std de los scalados {sc_data.mean(), sc_data.std()}")
    
scaler()