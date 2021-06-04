import numpy as np 
import torch

from time import time



def average(method):
    def averaged(*args, **kwargs):
        av_results = []
        n_trials = kwargs['n_trials']
        for trial in range(n_trials):
            if trial == 0:
                results = method(*args, **kwargs)
                for result in results[1:]:
                    av_results.append([result])
            else:
                results = method(*args, **kwargs)
                for ii, result in enumerate(results[1:]):
                    av_results[ii].append(result)
        return av_results 
    return averaged 
















    
    




