#Lab 4 Redes

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


"""
Funcion que crea el graficos de modulacion AM depediendo del porcentaje de
modulacion entregado. El porcentaje de modulación varía la amplitud de la onda
transportadora. Al finalizar la modulación, se realiza el llamado a la función
que realiza la demodulación
Entrada: T -> tiempo de la onda portadora (carrier, eje x)
         ondaInterpolada -> datos del mensaje (eje y)
         datosCarrier -> datos de la onda portadora (eje y)
         porcentaje -> porcentaje de modulación
         Modulador -> onda modulada AM (mensaje dentro de la onda portadora)
         fc -> frecuencia de la onda portadora

Salida:
      demo -> datos de la señal demodulada
                
"""

def graficarAM(T,ondaInterpolada,datosCarrier,porcentaje,Modulador,fc):
        porce = int(porcentaje*100)
        #Generando grafico Modulación AM
        plt.figure(1)
        plt.subplot(411)
        plt.title("Señal entrante") 
        plt.plot(T[0:1000],ondaInterpolada[0:1000])
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.subplot(412)
        plt.title("Señal portadora")
        plt.plot(T[0:1000],datosCarrier[0:1000],linewidth=0.3,color="red")
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.subplot(413)
        plt.title("Señal Modulada AM al " +str(porce)+"%")
        plt.plot(T[0:1000],Modulador[0:1000],linewidth=0.3,color = "green",marker = "o",markersize=0.5)
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        demo = demoduladorAM(porce,T,Modulador,fc)
        plt.subplot(414)
        plt.title("Señal Demodulada AM al " +str(porce)+"%")
        plt.plot(T[0:1000],demo[0:1000],linewidth=0.3,color="purple")
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        plt.show()
        return demo

"""
Funcion que crea graficos de modulacion FM, que al igual que la función anterior
se requiere el porcentaje de modulación pero en este caso, no afecta la amplitud
de la portadora, si no que a la frecuencia. No se realiza la demodulación de
esta función
Entrada:  T -> tiempo de la onda portadora (carrier, eje x)
         ondaInterpolada -> datos del mensaje (eje y)
         datosCarrier -> datos de la onda portadora (eje y)
         porcentaje -> porcentaje de modulación
         Modulador -> onda modulada AM (mensaje dentro de la onda portadora)
         fc -> frecuencia de la onda portadora
"""

def graficarFM(T,ondaInterpolada,datosCarrier,porcentaje,Modulador,fc):
        porce = int(porcentaje*100)
        #Generando grafico Modulacioin FM
        plt.figure(2)
        plt.subplot(311)
        plt.title("Señal entrante")
        plt.plot(T[:2000],ondaInterpolada[:2000],linewidth=0.3,color= "red")
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.subplot(312)
        plt.title("Señal portadora")
        plt.plot(T[:2000],datosCarrier[:2000],linewidth=0.3,color="red")
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.subplot(313)
        plt.title("Modulacion FM "+str(porce)+"%")
        plt.plot(T[:2000],Modulador[:2000],linewidth=0.3,color="green",marker ="o",markersize=0.5)
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        plt.show()
                    

"""
Funcion que realiza la demodulacion AM
Entrada: -porcentaje de modulacion
         -tiempo de la señal modulada
         -datos de la señal modulada
         -frecuencia del carrier

Salida: datos de la señal modulada
"""
def demoduladorAM(porcentaje,tiempoModulacion,datosM,fc):
        datosCarrier = porcentaje*cos(2*math.pi*fc*tiempoModulacion)
        demoduladorAM = datosM*datosCarrier
        return demoduladorAM


#Funcion que se intento para filtrar la demodulacion, pero no eliminaba el ruido
def lowFilter(datos,fs,fc):
    b,a = signal.butter(5,4000,btype='lowpass',fs=fc)
    filtro = signal.filtfilt(b,a,datos)
    return filtro


"""
Función que realiza la transformada de fourier en base a los datos obtenidos del audio.
Entrada:
        audio     -> señal en dominio del tiempo.
        fs     -> frecuencia de muestreo de la señal.
Salida:
        fftDatos  -> transformada de fourier normalizada para los valores de la señal original.
        fftFreq -> frecuencias de muestreo que dependen del largo del arreglo data y de rate.
""" 
def tFourier(audio,fs):
        n=len(audio)
        Ts = n/fs
        fftDatos = fft(audio) / n
        fftFreq = np.fft.fftfreq(n,1/fs)
        return(fftDatos,fftFreq)

#############################################
#Importando la señal de audio utilizando
#fs: Frencuencia del audio
#audio: arreglo con la informacion obtenida del audio wav
fs,audio = wavfile.read("handel.wav")

#Cantidad de intervalos del audio
intervalos=len(audio)

#Periodo maximo segun la frecuencia T= intevalos/frecuencia
Tmax = intervalos/fs

#Arreglo tiempo, distancia entre ellos
tiempo = np.linspace(0,Tmax,intervalos)

####Señal portadora#######
#Arreglo Delta T, se utiliza un valor grande para muestrear mas valores
muestreo = int(250000*Tmax)
T = np.linspace(0,Tmax,muestreo)

#Interpolación
#Se interpola para tener la misma cantidad de datos
ondaInterpolada = interp(T,tiempo,audio)

#Trnasformada de Fourier
fftAudio,fftFs = tFourier(audio,fs)

#Generando grafico del audio en funcion del tiempo
plt.figure(1)
plt.plot(tiempo,audio)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Grafico del audio Wav Original")
plt.show()

######################MODULADOR AM###################

#AM: y(t) = k*m(t)cos(2*pi*fc*t)
fc = 2*fs
k1 = 0.15
k2 = 1
k3 = 1.25


#Se calcula los datos de la onda portadora sin el mensaje, para cada porcentaje 
datosCarrier15 = (k1)*cos(2*math.pi*fc*T)
datosCarrier100 = (k2)*cos(2*math.pi*fc*T)
datosCarrier125 = (k3)*cos(2*math.pi*fc*T)

#Onda modula AM para cada porcentaje

moduladorAM_15 = datosCarrier15*ondaInterpolada
moduladorAM_100 = datosCarrier100*ondaInterpolada
moduladorAM_125 = datosCarrier125*ondaInterpolada

#Despues de graficar, la funcion devuelve la demodulación
demo15=graficarAM(T,ondaInterpolada,datosCarrier15,k1,moduladorAM_15,fc)
demo100=graficarAM(T,ondaInterpolada,datosCarrier100,k2,moduladorAM_100,fc)
demo125=graficarAM(T,ondaInterpolada,datosCarrier125,k3,moduladorAM_125,fc)

#se interpola demodulación
nuevaDemo15=interp(tiempo,T,demo15)
nuevaDemo100=interp(tiempo,T,demo100)
nuevaDemo125=interp(tiempo,T,demo125)



###CREAR AUDIOS WAV DE LOS DATOS DE LA SEÑAL CON MODULACION AM####
wavfile.write("salida_MODULACION_AM_"+str(k1*100)+".wav",fs,nuevaDemo15.astype('int16'))
wavfile.write("salida_MODULACION_AM "+str(k2*100)+".wav",fs,nuevaDemo100.astype('int16'))
wavfile.write("salida_MODULACION_AM "+str(k3*100)+".wav",fs,nuevaDemo125.astype('int16'))



######################MODULADOR FM###################
#Al igual que para AM se requiere el porcentaje de modulacion, pero en
#este caso, no afecta a la amplitudde la onda portadora , sino que a la freq.
#NO se realiza demodulacion de est funcion

#FM: y(t) = cos(2*pi*fc*t + k*integral(m*t))
fc = (5/2)*fs #Se necesita una frecuencia portadora que sea la mita de la freq obtenida y minimo 4 veces mayor que la frecuencia de muestreo
A=1 #Amplitud definida para la señal portadora
fc= 6500
#Se calcula los datos de la onda portadora sin el mensaje, para cada porcentaje
datosCarrier15 = A*cos(2*math.pi*fc*T)
datosCarrier100 = A*cos(2*math.pi*fc*T)
datosCarrier125 = A*cos(2*math.pi*fc*T)

#Se calcula la integral que contiene el mensaje
tiempo=T
Integral = integrate.cumtrapz(ondaInterpolada,tiempo,initial=0) #Integral acumulada

#Señal modulada en su frecuencia
moduladorFM_15 = A*cos(2*math.pi*fc*T + math.pi*k1*Integral) 
moduladorFM_100 = A*cos(2*math.pi*fc*T + math.pi*k2*Integral) 
moduladorFM_125 = A*cos(2*math.pi*fc*T + math.pi*k3*Integral)

#Obtener graficos de la señal modulada en FM
graficarFM(T,ondaInterpolada,datosCarrier15,k1,moduladorFM_15,fc)
graficarFM(T,ondaInterpolada,datosCarrier100,k2,moduladorFM_100,fc)
graficarFM(T,ondaInterpolada,datosCarrier125,k3,moduladorFM_125,fc)


nuevaDemo15FM=interp(tiempo,T,moduladorFM_15)
nuevaDemo100FM=interp(tiempo,T,moduladorFM_100)
nuevaDemo125FM=interp(tiempo,T,moduladorFM_125)

#Obtener la amplitud y la frecuencia mediante la funcion que usa transformada de
#Fourier
fftDatosAM15, fftFreqAM15 = tFourier(nuevaDemo15,fs)
fftDatosAM100, fftFreqAM100 = tFourier(nuevaDemo100,fs)
fftDatosAM125, fftFreqAM125 = tFourier(nuevaDemo125,fs)

fftDatosFM15, fftFreqFM15 = tFourier(nuevaDemo15FM,fs)
fftDatosFM100, fftFreqFM100 = tFourier(nuevaDemo100FM,fs)
fftDatosFM125, fftFreqFM125 = tFourier(nuevaDemo125FM,fs)


#Grafico Espectros de frecuencia

#Espectros de Frecuencia 15%
plt.figure(5)
plt.subplot(311)
plt.title("Señal Entrante")
plt.plot(fftFs,abs(fftAudio))
plt.subplot(312)
plt.title("Señal Modulada AM 15%")
plt.plot(fftFreqAM15,abs(fftDatosAM15))
plt.subplot(313)
plt.title("Señal Modulada FM 15%")
plt.plot(fftFreqFM15,abs(fftDatosFM15))
plt.tight_layout()
plt.show()

#Espectros de Frecuencia 100%
plt.figure(1)
plt.subplot(311)
plt.title("Señal Entrante")
plt.plot(fftFs,abs(fftAudio))
plt.subplot(312)
plt.title("Señal Modulada AM 100%")
plt.plot(fftFreqAM100,abs(fftDatosAM100))
plt.subplot(313)
plt.title("Señal Modulada FM 100%")
plt.plot(fftFreqFM100,abs(fftDatosFM100))
plt.tight_layout()
plt.show()

#Espectros de Frecuencia 125%
plt.figure(1)
plt.subplot(311)
plt.title("Señal Entrante")
plt.plot(fftFs,abs(fftAudio))
plt.subplot(312)
plt.title("Señal Modulada AM 125%")
plt.plot(fftFreqAM125,abs(fftDatosAM125))
plt.subplot(313)
plt.title("Señal Modulada FM 125%")
plt.plot(fftFreqFM125,abs(fftDatosFM125))
plt.tight_layout()
plt.show()
