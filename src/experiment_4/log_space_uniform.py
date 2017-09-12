# -*- coding: utf-8 -*-
"""
.. module:: log_space_uniform
   :platform: Unix, Windows
   :synopsis: This module contains a class used to generate random 
              numbers uniformly distributed in log space.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

from scipy.stats import rv_continuous
import numpy.random as mtrand

class LogSpaceUniformGen(rv_continuous):
    """Class used to generate random numbers uniformly distributed in
    log space.
    """
    def rvs(self,*args,**kwds):
        return 10**super().rvs(*args,**kwds)

    def _rvs(self):
        return mtrand.uniform(0.0,1.0,self._size)

    def _pdf(self, x):
        return 1.0*(x==x)
    
    def _cdf(self, x):
        return x
    
    def _ppf(self, q):
        return q
    
    def _stats(self):
        return 0.5, 1.0/12, 0, -1.2
    
    def _entropy(self):
        return 0.0
LogSpaceUniform = LogSpaceUniformGen(a=0.0, b=1.0, name='uniform')

def main():
    r = LogSpaceUniform(loc=0,scale=15)
    print(r.rvs(100))

if __name__ == "__main__":    
    main()