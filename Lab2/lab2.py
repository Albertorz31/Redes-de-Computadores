#Lab2 Redes

import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import scipy.signal as signal
from scipy.signal import lfilter, firwin, filtfilt
from scipy.signal import convolve as sig_convolve
import matplotlib.pyplot as plt
#Transformada de fourier
from scipy.fftpack import fft,fftfreq, ifft, fftshift

#Importando la señal de audio utilizando
#fs: Frencuencia del audio
#audio: arreglo con la informacion obtenida del audio wav
fs,audio = wavfile.read("handel.wav")

#Cantidad de intervalos del audio
intervalos= len(audio)

#Espectograma del audio original
plt.figure(1)
plt.specgram(audio, Fs=fs)
plt.colorbar()
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')

##########Aplicación de filtro FIR#############


#######Paso Bajo#######

b, a = signal.butter(5,0.1,btype='lowpass')
#b= vector de coeficiente numerador del filtro
#a= vector de coeficiente de denominador del filtro
#funcion signal.filfilt devuelve la salida filtrada de audio
filters=signal.filtfilt(b,a,audio)
lenA=len(audio)
tiempo=float(lenA)/fs
rate=np.arange(0,tiempo,1.0/fs)

#Se grafica la tranformada de fuourier inversa
#Se grafica el tiempo con la inversa de la tranformada
plt.figure(2)
plt.plot(rate,filters)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (Db)")
plt.title("Grafico transformada de Fourier Inversa con filtro paso bajo")

#Crear archivo wav
invfourier= np.asarray(filters,dtype=np.int16)
wavfile.write("AudioPasoBajo.wav",fs,invfourier)

#Espectograma después de aplicado el filtro paso bajo
plt.figure(3)
plt.specgram(filters, Fs=fs)
plt.colorbar()
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')

#######Paso Alto#######

b, a = signal.butter(5,0.7,btype='highpass')
#b= vector de coeficiente numerador del filtro
#a= vector de coeficiente de denominador del filtro
#funcion signal.filfilt devuelve la salida filtrada de audio
filters=signal.filtfilt(b,a,audio)
lenA=len(audio)
tiempo=float(lenA)/fs
rate=np.arange(0,tiempo,1.0/fs)

#Se grafica la tranformada de fuourier inversa
#Se grafica el tiempo con la inversa de la tranformada
plt.figure(4)
plt.plot(rate,filters)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (Db)")
plt.title("Grafico transformada de Fourier Inversa con filtro paso alto")

#Crear archivo wav
invfourier= np.asarray(filters,dtype=np.int16)
wavfile.write("AudioPasoAlto.wav",fs,invfourier)

#Espectograma después de aplicado el filtro paso bajo
plt.figure(5)
plt.specgram(filters, Fs=fs)
plt.colorbar()
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')

#######Paso Banda#######

b, a = signal.butter(5,[0.3,0.5],btype='bandpass')
#b= vector de coeficiente numerador del filtro
#a= vector de coeficiente de denominador del filtro
#funcion signal.filfilt devuelve la salida filtrada de audio
filters=signal.filtfilt(b,a,audio)
lenA=len(audio)
tiempo=float(lenA)/fs
rate=np.arange(0,tiempo,1.0/fs)

#Se grafica la tranformada de fuourier inversa
#Se grafica el tiempo con la inversa de la tranformada
plt.figure(6)
plt.plot(rate,filters)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (Db)")
plt.title("Grafico transformada de Fourier Inversa con filtro paso banda")

#Crear archivo wav
invfourier= np.asarray(filters,dtype=np.int16)
wavfile.write("AudioPasoBanda.wav",fs,invfourier)

#Espectograma después de aplicado el filtro paso bajo
plt.figure(7)
plt.specgram(filters, Fs=fs)
plt.colorbar()
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')


plt.show()
