# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:23:50 2020

@author: rashe
"""

from clustering import clusterGenerator
from data_loading import loadData
from curricula import curriculum

class generateCurriculum():
    
    def __init__(self, batches, outs, curric_type, numbatches=3, reverse=False):
        assert(type(batches) == dict)
        assert(type(outs) == dict)
        assert(len(batches) == len(outs))
        
        self.batches = batches
        self.outs = outs
        self.curric_type = curric_type
        self.reverse = reverse
        self.numbatches = numbatches
        self.cur = curriculum(numbatches=self.numbatches)
        
    def rankBatches(self):
        
        rankings = dict()
        for i in range(len(self.batches)):
            if self.curric_type.lower() == 'm':
                rankings[i] = self.cur.getMahalanobis(self.batches[i])
            elif self.curric_type.lower() == 'c':
                rankings[i] = self.cur.getCosine(self.batches[i])
            elif self.curric_type.lower() == 'w':
                rankings[i] = self.cur.getWasserstein(self.batches[i])
            
        return rankings
    
    def createBatches(self, cumulative=False, forward=True):
        curric_batches = dict()
        curric_outs = dict()
        rankings = self.rankBatches()
        for i in range(len(self.batches)):
        
            k = int(len(self.batches[i])/self.numbatches)
            if forward:
                if (self.curric_type.lower() == 'm' and forward) or (self.curric_type.lower() in ('c', 'w') and not forward) :
                    curric = rankings[i].sort_values(0)
                else:
                    curric = rankings[i].sort_values(0, ascending=False)
            
            ba = dict()
            ou = dict()
            for b in range(self.numbatches):
                if cumulative == True:
                    ba[b] = self.batches[i].iloc[curric[0:(b+1)*k].index]
                else:
                    ba[b] = self.batches[i].iloc[curric[b*k:(b+1)*k].index]
                ou[b] = self.outs[i].iloc[ba[b].index]
            curric_batches[i] = ba
            curric_outs[i] = ou
        
        return curric_batches, curric_outs
        
if __name__ == "__main__":
    
    data = loadData('digits') 
    cg = clusterGenerator(data,10, 3, multimodal=True, mix_type='gaussian') 
    b,o = cg.separateClusters() 
    gcu = generateCurriculum(b,o,curric_type='m',numbatches=10,reverse=False)
    curric_batches, curric_outs = gcu.createBatches(cumulative=True, forward=True)