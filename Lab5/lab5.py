#Lab 5 Redes

import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
from pylab import *
import scipy.integrate as integrate
import scipy.signal as signal
from scipy.fftpack import fft, ifft
from scipy.signal import firwin, lfilter, filtfilt
import matplotlib.pyplot as plt
import math
from random import randint, uniform,random

"""
Funcion que demodula segun la amplitud de la señal ingresada, si
el valor es 0 se ingresa un 0 a la lista, sino se agrega un 1
Entrada: Senial-> lista con las señales
Salida: salida_demodulada-> lista con los bits demodulados
"""
def demodulador(Senial):

  salida_Demodulada = []

  for x in Senial:
    if x == 0:
      salida_Demodulada.append(0)
    else:
      salida_Demodulada.append(1)
  
  return salida_Demodulada


"""
Funcion que modula con OOK la lista de bits, usa dos portadoras dependiendo del
bit (0 o 1), en cada portadora varia la amplitud, una de ellas la amplitud es 0
Entrada: listaBits -> bits iniciales
         A -> amplitud de la primera portadora
         B -> amplitud de la segunda portadora
         fc -> frecuencia de la onda portadora
         Tb -> tiempo de intervalo de cada bit en la portadora
Salida:
     salida_modulada -> lista con las señales
"""

def modulador( listaBits, A, B, fc, Tb):

  T=np.linspace(0,Tb,1000)

  portadora0 = A*cos(2*math.pi*fc*T)
  portadora1 = B*cos(2*math.pi*fc*T)

  salida_modulada = []

  for x in listaBits:
    if x == 0:
      salida_modulada.extend(portadora0)
    else:
      salida_modulada.extend(portadora1)

  return salida_modulada


#Se crea lista bits, la cual contiene los bits iniciales que se generan
#aleatoriamente
listaBits=[]

numero=randint(0,1)

for i in range(0,1000):
    listaBits.append(numero)
    numero=randint(0,1)



######################MODULADOR ASK###################
#0 = Acos(2*pi*fc*t)
#1 = Bcos(2*pi*fc*t)

#Si se realiza la modulación digital OOK, que es una modulación digital por
#encendido-apagado de portador, entonces la amplitud A seria igual a 0

A=0
B=7

#Frecuencia del carrier
fc=16384

#Bits 
Tb = 0.001


salida_modulada = modulador(listaBits, A, B, fc, Tb)

T=np.linspace(0,Tb*len(salida_modulada),1000000)

plt.figure(1)
plt.title("Señal")
plt.plot(T[0:10000],salida_modulada[0:10000],linewidth=0.5)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
#plt.show()

######################DEMODULADOR###################

salida_demodulada = demodulador(salida_modulada)

T=np.linspace(0,Tb*len(salida_demodulada),1000000)

plt.figure(2)
plt.title("Señal")
plt.plot(T[0:10000],salida_demodulada[0:10000],linewidth=0.7,color="red")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.show()






    
