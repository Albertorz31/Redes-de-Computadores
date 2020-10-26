import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
#Libreria scipy para la Transformada de fourier
from scipy.fftpack import fft,fftfreq,ifft


#Importando la señal de audio utilizando
#fs: Frencuencia del audio
#audio: arreglo con la informacion obtenida del audio wav
fs,audio = wavfile.read("handel.wav")

#Cantidad de intervalos del audio
intervalos= len(audio)

#Espaciado entre intervalos
dt= 1/fs

#Periodo maximo segun la frecuencia T= intevalos/frecuencia
Tmax = float(intervalos)/float(fs)
#Arreglo Delta T, distancia entre ellos
T=np.linspace(0,Tmax,intervalos) 

#Generando grafico
plt.figure(1)
plt.plot(T,audio)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Grafico del audio Wav")


#ftt: devuelve la tranformada de fourier
yfourier = fft(audio)/intervalos #normalizada
#fftfreq: devuelve la frecuencia de cada punto de la transformada de fourier
xfourier= fftfreq(int(intervalos),dt)


#Transformada de Fourier
plt.figure(2)
plt.plot(xfourier,abs(yfourier))
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Grafico transfromada de Fourier")


#Transformada de fourier inversa

#Se transforma la transfromada de fourier con la funcion ifft
yinvf= ifft(yfourier).real*intervalos

#Se grafica el tiempo con la inversa de la tranformada
plt.figure(3)
plt.plot(T,yinvf)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Grafico transfromada de Fourier Inversa")

#Obtencion del Componente de mayor amplitud
maximo= int(max(yfourier).real)

#Maximo es: 313,967
#Trunco la amplitud maxima al 15%
truncado = maximo*0.15


#Todos los valores de la amplitud menor al 15% maximo los igualo a 0
i=0
while i<len(yfourier):
    if(int(yfourier[i].real) < truncado):
        yfourier[i]=0
    i=i+1

plt.figure(4)
plt.plot(xfourier,abs(yfourier))
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Grafico transformada de fourier con el 15% de margen")


#Se transforma la transfromada de fourier con la funcion ifft
yinvf2= ifft(yfourier).real*intervalos

#Se grafica el tiempo con la inversa de la tranformada
plt.figure(5)
plt.plot(T,yinvf2)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Grafico de la transformada de fourier inversa con el 15%")
plt.show()

#Crear archivos .wav
invfourier= np.asarray(yinvf,dtype=np.int16)
yfiltrada = np.asarray(yinvf2,dtype=np.int16)

wavfile.write("AudioInversaTransformada.wav",fs,invfourier)
wavfile.write("AudioFiltrado.wav",fs,yfiltrada)

