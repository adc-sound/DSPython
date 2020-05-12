# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:28:07 2019

@author: Alessandro DE CECCO (ADC)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import Filters as F
from scipy.io import wavfile


from scipy import optimize



def amplify(x, A):
        
    """
    Parameters
    ----------
    x : Input signal
    A : Amplification factor

    Returns
    -------
    y : Output signal

    """
    # Init
    y = np.zeros(len(x))
    
    # Processor
    for i in range(len(x)):
        y[i] = A * x[i]
    return y

def apply_2ndorder_Filter_Generic(x,a,b):
    """
    Parameters
    ----------
    x : Input signal
    a, b : Filter coefficients 
    

    Returns
    -------
    y : Output signal

    """
    # Init
    if ((len(a) != 3 ) or (len(b) != 3)):
        raise ValueError("Ill-defined filter coefficients")
    
    y = np.zeros(len(x))
    
    
    # Reset filter states
    xz1 = 0
    xz2 = 0
    yz1 = 0
    yz2 = 0
    
    # Processor
    for i in range(len(x)):
       
        y[i] = (b[0] * x[i] + b[1] * xz1 + b[2] * xz2 - a[1] * yz1 - a[2]*yz2) / a[0]  # Direct Form I
        
        # Update filter states
        xz2 = xz1
        xz1 = x[i]
        yz2 = yz1
        yz1 = y[i]
      
    return y


def apply_1storder_Filter(x, Wc, G=0., filt_type='allpass'):
    """
    Parameters
    ----------
    x : Input signal
    Wc : Normalized cut-off frequency 0<Wc<1, i.e. 2*fc/fS.
    G: Gain in dB
    filt_type: filter type ('lowpass', 'highpass', 'allpass', 'lowshelf', 'highshelf')

    Returns
    -------
    y: Output signal

    """
    
    # Init
    y = np.zeros(len(x))
    if (filt_type=='lowpass' or filt_type=='highpass' or filt_type=='allpass'):
        G = 0.
    
    v0 = 10**(G/20.)  # linear gain
    H0 = v0 - 1. 
    c = (np.tan(np.pi*Wc/2.)-v0) / (np.tan(np.pi*Wc/2.)+v0)
        

    xz1 = 0.
    
    # Processor
    for i in range(len(x)):
        xz_new = x[i] - c*xz1
        ap_y = c * xz_new + xz1
        xz1 = xz_new
        if filt_type == 'allpass':
            y[i] = ap_y
        elif filt_type == 'lowpass':
            y[i] = 0.5 * (x[i] + ap_y)
        elif filt_type == 'highpass':
            y[i] = 0.5 * (x[i] - ap_y)
        elif filt_type == 'lowshelf':
            
        elif filt_type == 'highshelf':
            
        else:
            raise ValueError("Filter type not valid")
    return y


def apply_2ndorder_Filter(x, Wc, Wb, filt_type):
    """
    Parameters
    ----------
    x : Input signal
   
    Wc : Normalized center frequency 0<Wc<1, i.e. 2*fc/fS
    Wb: Normalized bandwidth  0<Wb<1, i.e. 2*fb/fS
    filt_type: filter type ('bandpass', 'bandreject', 'allpass')

    Returns
    -------
    y: Output signal

    """
    
    # Init
    y = np.zeros(len(x))

    c = (np.tan(np.pi*Wb/2.)-1.) / (np.tan(np.pi*Wb/2.)+1.)
    d = -np.cos(np.pi*Wc)
    xz1 = 0.
    xz2 = 0.
    
    # Processor
    for i in range(len(x)):
        xz_new = x[i] - d*(1-c)*xz1 + c*xz2
        ap_y = -c * xz_new + d*(1-c)*xz1 + xz2
        xz2 = xz1
        xz1 = xz_new
        
        if filt_type == 'allpass':
            y[i] = ap_y
        elif filt_type == 'bandpass':
            y[i] = 0.5 * (x[i] - ap_y)
        elif filt_type == 'bandreject':
            y[i] = 0.5 * (x[i] + ap_y)
        else:
            raise ValueError("Filter type not valid")
    return y
#%% Import and plot
filename = "test_dafx.wav"
fs, x = wavfile.read(filename)
x = x / (2**15 - 1)
n = np.arange(len(x))

# A = 0.5
# y = amplify(x,A)

a = [1, -1.28, 0.47]
b = [0.69, -1.38, 0.69]

fc = 400
fb = 100
y = apply_2ndorder_Filter(x+0.2, 2*fc/fs, 2*fb/fs,'allpass')
plt.figure()
plt.plot(n, x+0.2)
plt.plot(n, y)
plt.plot()
plt.xlim(0,len(x))
plt.xlabel("n")
plt.ylabel("x")
plt.grid()
plt.show()
