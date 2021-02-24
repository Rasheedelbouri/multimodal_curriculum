#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:42:53 2021

@author: kebl4170
"""
import matplotlib.pyplot as plt
import pathlib
import argparse
import ast

performances = {}

def loadResults(folder):
    for i,path in enumerate(pathlib.Path(folder).iterdir()):
        if path.is_file():
            f = open(path, "r")
            performances[i] = ast.literal_eval(f.read())
            f.close()
    
    return performances
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--curricType", type=str, 
                        help='Which curriculum')    
    parser.add_argument("--mixType", type=str, 
                        help='Number of curriculum batches')
    
    args = parser.parse_args()
    curricType = args.curricType
    mixType = args.mixType
    
    assert curricType in ('m', 'c', 'w')
    assert mixType in ("gaussian", "k-means", "class")
    
    performances = loadResults("../100rep_results/"+str(curricType)+"/"+str(mixType))
    for i in range(len(performances)):
        plt.plot(performances[i])
    plt.show()