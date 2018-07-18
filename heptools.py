# %load heptools.py
#!/usr/bin/env python
#DEBUG: Fix casas_ibarra test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
import pylab
import cmath
import math
import commands
import sys
import time
import re
import os
# %load configrun_IDM.py
def get_config(key):
    import numpy as np
    import commands
    import sys
    
    if key=='SM':
        return {'G_F': 1.166e-05, 'alpha_em': 0.0078125, 'm_e': 0.000511, \
                'm_mu': 0.1057, 'm_tau': 1.777, 'vev': 246.0, \
                'MW': 80.385, 'MZ': 91.1876, 'WH':4.21e-3,'m_pi0':134.9766E-3,\
                'm_pipm':139.57018E-3,'hbarc':1.973269631E-16,'V_ud':0.97425,\
                'f_pi':131E-3}


    
        #test constants

def Lambda(Mk,mH0,mA0):
    '''Loop funtion, defined for each neutral triplet component'''
    return (Mk/(16*pi**2))*((mH0**2/(mH0**2-Mk**2))*np.log(mH0**2/Mk**2) -(mA0**2/(mA0**2-Mk**2))*np.log(mA0**2/Mk**2)  )


def neutrino_data(CL=3,IH=False):
    import numpy as np
    '''From arxiv:1405.7540 (table I)
    and asumming a Normal Hierarchy:
    Output:
    mnu1in: laightest neutrino mass
    Dms2: \Delta m^2_{12}
    Dma2: \Delta m^2_{13}
    ThetSol,ThetAtm,ThetRec: in radians
    '''
    if CL==3:
        Dms2=np.array([7.11e-5, 7.60e-5, 8.18e-5])*1e-18 # In GeV
        Dma2=np.array([2.30e-3, 2.48e-3, 2.65e-3])*1e-18 # In GeV
        if IH:
            Dma2=np.array([2.20e-3, 2.38e-3, 2.54e-3])*1e-18 # In GeV
        #input real values:
        #
        ThetSol = np.array([0.278,  0.323,  0.375]) 
        ThetAtm = np.array([0.393,  0.567,  0.643])
        if IH:
            ThetAtm = np.array([0.403,  0.573,  0.640])
        ThetRec = np.array([0.0190, 0.0226, 0.0262])
        if IH:
            ThetRec = np.array([0.0193, 0.0229, 0.0265])
            
        delta=np.array([0,np.pi,2.*np.pi])
    elif CL==1:
        Dms2=np.array([7.11e-5, 7.60e-5, 8.18e-5])*1e-18 # In GeV
        Dma2=np.array([2.41e-3, 2.48e-3, 2.53e-3])*1e-18 # In GeV
        if IH:
            Dma2=np.array([2.32e-3, 2.38e-3, 2.43e-3])*1e-18 # In GeV
        #input real values:
        #
        ThetSol = np.array([0.307,  0.323,  0.339]) 
        ThetAtm = np.array([0.443,  0.567,  0.599])
        if IH:
            ThetAtm = np.array([0.534,  0.573,  0.598])
        ThetRec = np.array([0.0214, 0.0226, 0.0238])
        if IH:
            ThetRec = np.array([0.0217, 0.0229, 0.0241])
        delta=np.array([1.01*np.pi,1.41*np.pi,1.96*np.pi])
        if IH:
            delta=np.array([1.17*np.pi,1.48*np.pi,1.79*np.pi])
    mnu1in=1E-5*1E-9

    return mnu1in,Dms2,Dma2,ThetSol,ThetAtm,ThetRec,delta

def casasibarra(di,norotate1=False,norotate2=False,norotate3=False,bestfit=False,\
                nophases=False,massless_nulight=False,min_nulight=1E-9,max_nulight=0.5,\
                IH=False,R_complex=True,min_angle=0,CL=3):
    """
    min_nulight=1E-9,
    max_nulight=0.5 in eV: pdg neutrino review
       NH: nu1<nu2<nu3
    We assume mass ordering for Heavy particles but inverse hierarchy (IH) for neutrinos could imply:
       IH: nu3<nu1<nu2
    di.keys()-> ['MH0','MA0','Mtr01','Mtr02','Mtr03',]
    """
    import numpy as np
    import pandas as pd
    di=pd.Series(di)
    if massless_nulight:
        norotate2=True #R13=1
        if not IH:
            norotate1=True #R12=1
        else:
            norotate3=True #R23=1
        
        
    
    ignore,Dms2,Dma2,ThetSol,ThetAtm,ThetRec,deltaD=neutrino_data(CL=CL,IH=IH) 

    
    #Inverse MR masses. M^R_3 -> infty corresponds to zero entry
    Mtr01t=1./np.abs(Lambda(di.Mtr01,di.MH0,di.MA0))
    Mtr02t=1./np.abs(Lambda(di.Mtr02,di.MH0,di.MA0))
    Mtr03t=1.*( 0. if massless_nulight else 1./np.abs(Lambda(di.Mtr03,di.MH0,di.MA0)) )
    
    
    DMR=np.asarray([ [np.sqrt( Mtr01t),0,0],[0,np.sqrt(Mtr02t),0],[0,0,np.sqrt(Mtr03t)] ]   )
    
    #phases of the PMNS matrix
    
    delta=1.*(0 if nophases else np.random.uniform(deltaD[0],deltaD[2]))
    eta1 =1.*(0 if nophases else np.random.uniform(0.,np.pi)) 
    eta2 =1.*(0 if nophases else np.random.uniform(0.,np.pi))
    if bestfit:
        delta=deltaD[1]
    
    if not IH:
        mnu1=np.exp(np.random.uniform(np.log(min_nulight),np.log(max_nulight)))*1e-9 
        #m_3=m_3^2-m_1^2+m_1^2
        mnu3=1.*(np.sqrt(Dma2[1]+mnu1**2)     if bestfit else  np.sqrt(np.random.uniform(Dma2[0],Dma2[2]) + mnu1**2) )
    else:
        mnu3=np.exp(np.random.uniform(np.log(min_nulight),np.log(max_nulight)))*1e-9 
        #m_1=|m_3^2-m_1^2|+m_3^2=m_1^2-m_3^2+m_3^2=
        mnu1=1.*(np.sqrt(Dma2[1]+mnu3**2)     if bestfit else  np.sqrt(np.random.uniform(Dma2[0],Dma2[2]) + mnu3**2) )

    #m_2=m_2^2-m_1^2+m_1^2    
    mnu2=1.*(np.sqrt(Dms2[1]+mnu1**2)     if bestfit else  np.sqrt(np.random.uniform(Dms2[0],Dms2[2]) + mnu1**2) ) 

    if massless_nulight:
        if IH:
            mnu3=0
        else:
            mnu1=0   

    #light neutrino masses only for an estimation 
    #mnu1=0
    #mnu2=sqrt(8.2e-5*1e-18+mnu1**2)
    #mnu3=sqrt(2.74e-3*1e-18+mnu1**2)
    
    #Square root of left-handed nuetrino mass matrix 
    DMnu=np.asarray([ [np.sqrt(mnu1),0,0],[0,np.sqrt(mnu2),0],[0,0,np.sqrt(mnu3)] ])
    
    #mixing angles using 3 sigma data from arxiv:1405.7540 (table I)                        
    #and asumming a Normal Hierarchy'''



    t12 = 1.*( np.arcsin(np.sqrt(ThetSol[1])) if bestfit else np.arcsin(np.sqrt(np.random.uniform(ThetSol[0],ThetSol[2]))))
    t23 = 1.*( np.arcsin(np.sqrt(ThetAtm[1])) if bestfit else np.arcsin(np.sqrt(np.random.uniform(ThetAtm[0],ThetAtm[2]))))
    t13 = 1.*( np.arcsin(np.sqrt(ThetRec[1])) if bestfit else np.arcsin(np.sqrt(np.random.uniform(ThetRec[0],ThetRec[2]))))
    
    
    #Building PMNS matrix: http://pdg.lbl.gov/2014/reviews/rpp2014-rev-neutrino-mixing.pdf
    
    U12 = np.array([ [np.cos(t12),np.sin(t12),0], [-np.sin(t12),np.cos(t12),0], [0,0,1.0] ])
    U13 = np.array([ [np.cos(t13),0,np.sin(t13)*np.exp(-delta*1j)], [0,1.0,0],\
                     [-np.sin(t13)*np.exp(delta*1j),0,np.cos(t13)] ])
    U23 = np.array([ [1.0,0,0], [0,np.cos(t23),np.sin(t23)], [0,-np.sin(t23),np.cos(t23)] ])
    Uphases = np.diag([1.,np.exp(eta1*1j/2.),np.exp(eta2*1j/2.)])
    U=((U23.dot(U13)).dot(U12)).dot(Uphases)
    #print U-np.dot(U23,np.dot(U13,np.dot(U12,Uphases)))
    #Building R matrix of the Casas-Ibarra parametrization
    
    
    min_real=min_angle
    max_real=2.*np.pi
    phases2=np.random.uniform(min_real,max_real ,3) 
    if R_complex:
        min_imag=1E-12
        max_imag=20. #2E-1
        phases2=phases2+1j*np.exp(np.random.uniform(\
                            np.log(min_imag),np.log(max_imag) ,3) )*np.random.choice([1,-1])
    

    b12 = 1.*(0 if norotate1 else phases2[0])
    b13 = 1.*(0 if norotate2 else phases2[1])
    b23 = 1.*(0 if norotate3 else phases2[2])
    
  
    # R 
    R12 = array([ [np.cos(b12),np.sin(b12),0], [-np.sin(b12),np.cos(b12),0], [0,0,1.0] ])
    R13 = array([ [np.cos(b13),0,np.sin(b13)], [0,1.0,0], [-np.sin(b13),0,np.cos(b13)] ])
    R23 = array([ [1.0,0,0], [0,np.cos(b23),np.sin(b23)], [0,-np.sin(b23),np.cos(b23)] ])
    R=dot(R23,dot(R13,R12))
    #1assert(np.abs(np.dot(R, R.transpose()))[1,2]<1E-10)
    #Yukawa matrix of the Casas-Ibarra parametrization
    yuk2=dot(DMR,dot(R,dot(DMnu,transpose(conjugate(U)))))
    return yuk2,U,DMnu.dot(DMnu),phases2


def Fme(x,xmin=0.996,xmax=1.005,xfit=1.001):
    """Fixing near to one values
     xmin: close to 1 from below
     xmax: close to 1 from above
     xfit:  optimized 1 limit
    """
    x=np.asarray(x)
    if x.shape:
        x[np.logical_and(x>xmin,x<xmax)]=xfit
    else:
        if x>xmin and x<xmax:
            x=xfit
        
    return (1.-6.*x+3.*x**2+2.*x**3-6*x**2*np.log(x))/(6.*(x-1.)**4)

def LFV(SM,dp,yuk):
    '''Oscar Notes with \pi^2 -> \pi
    '''
    Brmuegmax=5.7e-13
    Brtaumugmax=4.5e-8
    import numpy as np
    const=False
    SM=pd.Series(SM)
    y=np.matrix(yuk)
    
    dp=pd.Series(dp)
    FMl=[]
    for i in range(1,4):
        dp['Mtrp%d' %i]=dp['Mtr0%d' %i] #degenerate fermions
        FMl.append(1./(dp.MHC**2)*Fme(dp['Mtr0%d' %i]**2/dp.MHC**2)\
                   -1./(dp['Mtr0%d' %i]**2)*( Fme(dp.MH0**2/dp['Mtrp%d' %i]**2)\
                                            + Fme(dp.MA0**2/dp['Mtrp%d' %i]**2)   )  )
    FM=np.matrix(np.diag(FMl))
    
    Brmueg  =(3.*SM.alpha_em/(4.*16.*np.pi*SM.G_F**2)*np.abs(y[:,0].T*FM*y[:,1].conjugate())**2)[0,0]
    Brtaumug=(3.*SM.alpha_em/(4.*16.*np.pi*SM.G_F**2)*np.abs(y[:,1].T*FM*y[:,2].conjugate())**2)[0,0]
    if (Brmueg<Brmuegmax):
       if (Brtaumug<Brtaumugmax):
           const=True

    return Brmueg,Brtaumug,const
        

