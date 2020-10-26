import scipy
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image



#########Lectura de una imagen#############
path="leena512.bmp"
img= Image.open(path)
f=  imageio.imread('leena512.bmp')
type(f)


#Obtencion de la transformada de Fourier de la imagen
plt.figure(1)
fourier= np.fft.fft2(f)
fshift = np.fft.fftshift(fourier)
magnitudFFT= 20*np.log(np.abs(fshift))
#Graficar  la transformada de Fourier de la imagen
plt.xlabel("Ancho")
plt.ylabel("Largo")
plt.title("Transformada de Fourier de la imagen original")
plt.imshow(magnitudFFT)

#Largo y ancho de la imagen
lista=img.size
largoF=len(f)
width=lista[0]
height=lista[1]

#Listas de los colores RGB, osea, rojo,verde y azul. Estas lista es donde se guardaran
#los colores del pixel de la convolucion .
rojo=[0]*25
rojoBordes = [0]*25
verde = [0]*25
verdeBordes = [0]*25
azul = [0]*25
azulBordes = [0]*25

##############Kernel####################
#Se tiene 2 kernel, el de suavizado y el kernel de bordes
kernelSuavizado=[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
kernelBordes = [[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]]
ancho=len(kernelSuavizado)
largo=len(kernelSuavizado[0])

for i in range(0,ancho-1):
    for j in range(0,largo-1):
        kernelSuavizado[i][j]=kernelSuavizado[i][j]*(1/256)


#Se crea nueva imagen donde se iran almacenando los pixeles
#resultantes de la convolucion 
newImagSuavizado = Image.new("RGB",(width,height),"white")
newImagBordes = Image.new("RGB",(width,height),"white")


#Buble donde se realiza la convolucion entre la imagen inicial y el kernel
#se va guardando los valores de los colores en las listas para despues sumaralas
#y guardar el valor de ese pixel
for i in range(0,width):
    for j in range(0,height):
        #Si el pixel es de borde o sus valores porixmos no estan
        #dentro de la matriz, entonces se rempleza por 0
        if  i <= 2 or j <= 2 or i>= width-2 or j>= height -2:
            newImagSuavizado.putpixel((i,j),(0,0,0))
            newImagBordes.putpixel((i,j),(0,0,0))

        else:
            rojo[0] = (int)(img.getpixel((i-2,j-2))*kernelSuavizado[0][0])
            rojo[1] = (int)(img.getpixel((i-2,j-1))*kernelSuavizado[0][1])
            rojo[2] = (int)(img.getpixel((i-2,j))*kernelSuavizado[0][2])
            rojo[3] = (int)(img.getpixel((i-2,j+1))*kernelSuavizado[0][3])
            rojo[4] = (int)(img.getpixel((i-2,j+2))*kernelSuavizado[0][4])
            rojo[5] = (int)(img.getpixel((i-1,j-2))*kernelSuavizado[1][0])
            rojo[6] = (int)(img.getpixel((i-1,j-1))*kernelSuavizado[1][1])
            rojo[7] = (int)(img.getpixel((i-1,j))*kernelSuavizado[1][2])
            rojo[8] = (int)(img.getpixel((i-1,j+1))*kernelSuavizado[1][3])
            rojo[9] = (int)(img.getpixel((i-1,j+2))*kernelSuavizado[1][4])
            rojo[10] = (int)(img.getpixel((i,j-2))*kernelSuavizado[2][0])
            rojo[11] = (int)(img.getpixel((i,j-1))*kernelSuavizado[2][1])
            rojo[12] = (int)(img.getpixel((i,j))*kernelSuavizado[2][2])
            rojo[13] = (int)(img.getpixel((i,j+1))*kernelSuavizado[2][3])
            rojo[14] = (int)(img.getpixel((i,j+2))*kernelSuavizado[2][4])
            rojo[15] = (int)(img.getpixel((i+1,j-2))*kernelSuavizado[3][0])
            rojo[16] = (int)(img.getpixel((i+1,-1))*kernelSuavizado[3][1])
            rojo[17] = (int)(img.getpixel((i+1,j))*kernelSuavizado[3][2])
            rojo[18] = (int)(img.getpixel((i+1,j+1))*kernelSuavizado[3][3])
            rojo[19] = (int)(img.getpixel((i+1,j+2))*kernelSuavizado[3][4])
            rojo[20] = (int)(img.getpixel((i+2,j-2))*kernelSuavizado[4][0])
            rojo[21] = (int)(img.getpixel((i+2,j-1))*kernelSuavizado[4][1])
            rojo[22] = (int)(img.getpixel((i+2,j))*kernelSuavizado[4][2])
            rojo[23] = (int)(img.getpixel((i+2,j+1))*kernelSuavizado[4][3])
            rojo[24] = (int)(img.getpixel((i+2,j+2))*kernelSuavizado[4][4])

            rojoBordes[0] = (int)(img.getpixel((i-2,j-2))*kernelBordes[0][0])
            rojoBordes[1] = (int)(img.getpixel((i-2,j-1))*kernelBordes[0][1])
            rojoBordes[2] = (int)(img.getpixel((i-2,j))*kernelBordes[0][2])
            rojoBordes[3] = (int)(img.getpixel((i-2,j+1))*kernelBordes[0][3])
            rojoBordes[4] = (int)(img.getpixel((i-2,j+2))*kernelBordes[0][4])
            rojoBordes[5] = (int)(img.getpixel((i-1,j-2))*kernelBordes[1][0])
            rojoBordes[6] = (int)(img.getpixel((i-1,j-1))*kernelBordes[1][1])
            rojoBordes[7] = (int)(img.getpixel((i-1,j))*kernelBordes[1][2])
            rojoBordes[8] = (int)(img.getpixel((i-1,j+1))*kernelBordes[1][3])
            rojoBordes[9] = (int)(img.getpixel((i-1,j+2))*kernelBordes[1][4])
            rojoBordes[10] = (int)(img.getpixel((i,j-2))*kernelBordes[2][0])
            rojoBordes[11] = (int)(img.getpixel((i,j-1))*kernelBordes[2][1])
            rojoBordes[12] = (int)(img.getpixel((i,j))*kernelBordes[2][2])
            rojoBordes[13] = (int)(img.getpixel((i,j+1))*kernelBordes[2][3])
            rojoBordes[14] = (int)(img.getpixel((i,j+2))*kernelBordes[2][4])
            rojoBordes[15] = (int)(img.getpixel((i+1,j-2))*kernelBordes[3][0])
            rojoBordes[16] = (int)(img.getpixel((i+1,-1))*kernelBordes[3][1])
            rojoBordes[17] = (int)(img.getpixel((i+1,j))*kernelBordes[3][2])
            rojoBordes[18] = (int)(img.getpixel((i+1,j+1))*kernelBordes[3][3])
            rojoBordes[19] = (int)(img.getpixel((i+1,j+2))*kernelBordes[3][4])
            rojoBordes[20] = (int)(img.getpixel((i+2,j-2))*kernelBordes[4][0])
            rojoBordes[21] = (int)(img.getpixel((i+2,j-1))*kernelBordes[4][1])
            rojoBordes[22] = (int)(img.getpixel((i+2,j))*kernelBordes[4][2])
            rojoBordes[23] = (int)(img.getpixel((i+2,j+1))*kernelBordes[4][3])
            rojoBordes[24] = (int)(img.getpixel((i+2,j+2))*kernelBordes[4][4])

            verde[0] = (int)(img.getpixel((i-2,j-2))*kernelSuavizado[0][0])
            verde[1] = (int)(img.getpixel((i-2,j-1))*kernelSuavizado[0][1])
            verde[2] = (int)(img.getpixel((i-2,j))*kernelSuavizado[0][2])
            verde[3] = (int)(img.getpixel((i-2,j+1))*kernelSuavizado[0][3])
            verde[4] = (int)(img.getpixel((i-2,j+2))*kernelSuavizado[0][4])
            verde[5] = (int)(img.getpixel((i-1,j-2))*kernelSuavizado[1][0])
            verde[6] = (int)(img.getpixel((i-1,j-1))*kernelSuavizado[1][1])
            verde[7] = (int)(img.getpixel((i-1,j))*kernelSuavizado[1][2])
            verde[8] = (int)(img.getpixel((i-1,j+1))*kernelSuavizado[1][3])
            verde[9] = (int)(img.getpixel((i-1,j+2))*kernelSuavizado[1][4])
            verde[10] = (int)(img.getpixel((i,j-2))*kernelSuavizado[2][0])
            verde[11] = (int)(img.getpixel((i,j-1))*kernelSuavizado[2][1])
            verde[12] = (int)(img.getpixel((i,j))*kernelSuavizado[2][2])
            verde[13] = (int)(img.getpixel((i,j+1))*kernelSuavizado[2][3])
            verde[14] = (int)(img.getpixel((i,j+2))*kernelSuavizado[2][4])
            verde[15] = (int)(img.getpixel((i+1,j-2))*kernelSuavizado[3][0])
            verde[16] = (int)(img.getpixel((i+1,-1))*kernelSuavizado[3][1])
            verde[17] = (int)(img.getpixel((i+1,j))*kernelSuavizado[3][2])
            verde[18] = (int)(img.getpixel((i+1,j+1))*kernelSuavizado[3][3])
            verde[19] = (int)(img.getpixel((i+1,j+2))*kernelSuavizado[3][4])
            verde[20] = (int)(img.getpixel((i+2,j-2))*kernelSuavizado[4][0])
            verde[21] = (int)(img.getpixel((i+2,j-1))*kernelSuavizado[4][1])
            verde[22] = (int)(img.getpixel((i+2,j))*kernelSuavizado[4][2])
            verde[23] = (int)(img.getpixel((i+2,j+1))*kernelSuavizado[4][3])
            verde[24] = (int)(img.getpixel((i+2,j+2))*kernelSuavizado[4][4])

            verdeBordes[0] = (int)(img.getpixel((i-2,j-2))*kernelBordes[0][0])
            verdeBordes[1] = (int)(img.getpixel((i-2,j-1))*kernelBordes[0][1])
            verdeBordes[2] = (int)(img.getpixel((i-2,j))*kernelBordes[0][2])
            verdeBordes[3] = (int)(img.getpixel((i-2,j+1))*kernelBordes[0][3])
            verdeBordes[4] = (int)(img.getpixel((i-2,j+2))*kernelBordes[0][4])
            verdeBordes[5] = (int)(img.getpixel((i-1,j-2))*kernelBordes[1][0])
            verdeBordes[6] = (int)(img.getpixel((i-1,j-1))*kernelBordes[1][1])
            verdeBordes[7] = (int)(img.getpixel((i-1,j))*kernelBordes[1][2])
            verdeBordes[8] = (int)(img.getpixel((i-1,j+1))*kernelBordes[1][3])
            verdeBordes[9] = (int)(img.getpixel((i-1,j+2))*kernelBordes[1][4])
            verdeBordes[10] = (int)(img.getpixel((i,j-2))*kernelBordes[2][0])
            verdeBordes[11] = (int)(img.getpixel((i,j-1))*kernelBordes[2][1])
            verdeBordes[12] = (int)(img.getpixel((i,j))*kernelBordes[2][2])
            verdeBordes[13] = (int)(img.getpixel((i,j+1))*kernelBordes[2][3])
            verdeBordes[14] = (int)(img.getpixel((i,j+2))*kernelBordes[2][4])
            verdeBordes[15] = (int)(img.getpixel((i+1,j-2))*kernelBordes[3][0])
            verdeBordes[16] = (int)(img.getpixel((i+1,-1))*kernelBordes[3][1])
            verdeBordes[17] = (int)(img.getpixel((i+1,j))*kernelBordes[3][2])
            verdeBordes[18] = (int)(img.getpixel((i+1,j+1))*kernelBordes[3][3])
            verdeBordes[19] = (int)(img.getpixel((i+1,j+2))*kernelBordes[3][4])
            verdeBordes[20] = (int)(img.getpixel((i+2,j-2))*kernelBordes[4][0])
            verdeBordes[21] = (int)(img.getpixel((i+2,j-1))*kernelBordes[4][1])
            verdeBordes[22] = (int)(img.getpixel((i+2,j))*kernelBordes[4][2])
            verdeBordes[23] = (int)(img.getpixel((i+2,j+1))*kernelBordes[4][3])
            verdeBordes[24] = (int)(img.getpixel((i+2,j+2))*kernelBordes[4][4])

            azul[0] = (int)(img.getpixel((i-2,j-2))*kernelSuavizado[0][0])
            azul[1] = (int)(img.getpixel((i-2,j-1))*kernelSuavizado[0][1])
            azul[2] = (int)(img.getpixel((i-2,j))*kernelSuavizado[0][2])
            azul[3] = (int)(img.getpixel((i-2,j+1))*kernelSuavizado[0][3])
            azul[4] = (int)(img.getpixel((i-2,j+2))*kernelSuavizado[0][4])
            azul[5] = (int)(img.getpixel((i-1,j-2))*kernelSuavizado[1][0])
            azul[6] = (int)(img.getpixel((i-1,j-1))*kernelSuavizado[1][1])
            azul[7] = (int)(img.getpixel((i-1,j))*kernelSuavizado[1][2])
            azul[8] = (int)(img.getpixel((i-1,j+1))*kernelSuavizado[1][3])
            azul[9] = (int)(img.getpixel((i-1,j+2))*kernelSuavizado[1][4])
            azul[10] = (int)(img.getpixel((i,j-2))*kernelSuavizado[2][0])
            azul[11] = (int)(img.getpixel((i,j-1))*kernelSuavizado[2][1])
            azul[12] = (int)(img.getpixel((i,j))*kernelSuavizado[2][2])
            azul[13] = (int)(img.getpixel((i,j+1))*kernelSuavizado[2][3])
            azul[14] = (int)(img.getpixel((i,j+2))*kernelSuavizado[2][4])
            azul[15] = (int)(img.getpixel((i+1,j-2))*kernelSuavizado[3][0])
            azul[16] = (int)(img.getpixel((i+1,-1))*kernelSuavizado[3][1])
            azul[17] = (int)(img.getpixel((i+1,j))*kernelSuavizado[3][2])
            azul[18] = (int)(img.getpixel((i+1,j+1))*kernelSuavizado[3][3])
            azul[19] = (int)(img.getpixel((i+1,j+2))*kernelSuavizado[3][4])
            azul[20] = (int)(img.getpixel((i+2,j-2))*kernelSuavizado[4][0])
            azul[21] = (int)(img.getpixel((i+2,j-1))*kernelSuavizado[4][1])
            azul[22] = (int)(img.getpixel((i+2,j))*kernelSuavizado[4][2])
            azul[23] = (int)(img.getpixel((i+2,j+1))*kernelSuavizado[4][3])
            azul[24] = (int)(img.getpixel((i+2,j+2))*kernelSuavizado[4][4])

            azulBordes[0] = (int)(img.getpixel((i-2,j-2))*kernelBordes[0][0])
            azulBordes[1] = (int)(img.getpixel((i-2,j-1))*kernelBordes[0][1])
            azulBordes[2] = (int)(img.getpixel((i-2,j))*kernelBordes[0][2])
            azulBordes[3] = (int)(img.getpixel((i-2,j+1))*kernelBordes[0][3])
            azulBordes[4] = (int)(img.getpixel((i-2,j+2))*kernelBordes[0][4])
            azulBordes[5] = (int)(img.getpixel((i-1,j-2))*kernelBordes[1][0])
            azulBordes[6] = (int)(img.getpixel((i-1,j-1))*kernelBordes[1][1])
            azulBordes[7] = (int)(img.getpixel((i-1,j))*kernelBordes[1][2])
            azulBordes[8] = (int)(img.getpixel((i-1,j+1))*kernelBordes[1][3])
            azulBordes[9] = (int)(img.getpixel((i-1,j+2))*kernelBordes[1][4])
            azulBordes[10] = (int)(img.getpixel((i,j-2))*kernelBordes[2][0])
            azulBordes[11] = (int)(img.getpixel((i,j-1))*kernelBordes[2][1])
            azulBordes[12] = (int)(img.getpixel((i,j))*kernelBordes[2][2])
            azulBordes[13] = (int)(img.getpixel((i,j+1))*kernelBordes[2][3])
            azulBordes[14] = (int)(img.getpixel((i,j+2))*kernelBordes[2][4])
            azulBordes[15] = (int)(img.getpixel((i+1,j-2))*kernelBordes[3][0])
            azulBordes[16] = (int)(img.getpixel((i+1,-1))*kernelBordes[3][1])
            azulBordes[17] = (int)(img.getpixel((i+1,j))*kernelBordes[3][2])
            azulBordes[18] = (int)(img.getpixel((i+1,j+1))*kernelBordes[3][3])
            azulBordes[19] = (int)(img.getpixel((i+1,j+2))*kernelBordes[3][4])
            azulBordes[20] = (int)(img.getpixel((i+2,j-2))*kernelBordes[4][0])
            azulBordes[21] = (int)(img.getpixel((i+2,j-1))*kernelBordes[4][1])
            azulBordes[22] = (int)(img.getpixel((i+2,j))*kernelBordes[4][2])
            azulBordes[23] = (int)(img.getpixel((i+2,j+1))*kernelBordes[4][3])
            azulBordes[24] = (int)(img.getpixel((i+2,j+2))*kernelBordes[4][4])

            rojoFinalSuavizado = ((rojo[0] + rojo[1] + rojo[2] + rojo[3] + rojo[4] + rojo[5]
                     + rojo[6] + rojo[7] + rojo[8] + rojo[9] + rojo[10] + rojo[11] + 
                      rojo[12] + rojo[13] + rojo[14] + rojo[15] + rojo[16] + rojo[17] + 
                      rojo[18] + rojo[19] + rojo[20] + rojo[21] + rojo[22] + rojo[23] + rojo[24])//25)

            verdeFinalSuavizado = ((verde[0] + verde[1] + verde[2] + verde[3] + verde[4] + verde[5]
                     + verde[6] + verde[7] + verde[8] + verde[9] + verde[10] + verde[11] + 
                      verde[12] + verde[13] + verde[14] + verde[15] + verde[16] + verde[17] + 
                      verde[18] + verde[19] + verde[20] + verde[21] + verde[22] + verde[23] + verde[24])//25)

            azulFinalSuavizado = ((azul[0] + azul[1] + azul[2] + azul[3] + azul[4] + azul[5]
                     + azul[6] + azul[7] + azul[8] + azul[9] + azul[10] + azul[11] + 
                      azul[12] + azul[13] + azul[14] + azul[15] + azul[16] + azul[17] + 
                      azul[18] + azul[19] + azul[20] + azul[21] + azul[22] + azul[23] + azul[24])//25)

            rojoFinalBordes = ((rojoBordes[0] + rojoBordes[1] + rojoBordes[2] + rojoBordes[3] + rojoBordes[4] + rojoBordes[5]
                     + rojoBordes[6] + rojoBordes[7] + rojoBordes[8] + rojoBordes[9] + rojoBordes[10] + rojoBordes[11] + 
                      rojoBordes[12] + rojoBordes[13] + rojoBordes[14] + rojoBordes[15] + rojoBordes[16] + rojoBordes[17] + 
                      rojoBordes[18] + rojoBordes[19] + rojoBordes[20] + rojoBordes[21] + rojoBordes[22] + rojoBordes[23] + rojoBordes[24])//25)

            verdeFinalBordes = ((verdeBordes[0] + verdeBordes[1] + verdeBordes[2] + verdeBordes[3] + verdeBordes[4] + verdeBordes[5]
                     + verdeBordes[6] + verdeBordes[7] + verdeBordes[8] + verdeBordes[9] + verdeBordes[10] + verdeBordes[11] + 
                      verdeBordes[12] + verdeBordes[13] + verdeBordes[14] + verdeBordes[15] + verdeBordes[16] + verdeBordes[17] + 
                      verdeBordes[18] + verdeBordes[19] + verdeBordes[20] + verdeBordes[21] + verdeBordes[22] + verdeBordes[23] + verdeBordes[24])//25)

            azulFinalBordes = ((azulBordes[0] + azulBordes[1] + azulBordes[2] + azulBordes[3] + azulBordes[4] + azulBordes[5]
                     + azulBordes[6] + azulBordes[7] + azulBordes[8] + azulBordes[9] + azulBordes[10] + azulBordes[11] + 
                      azulBordes[12] + azulBordes[13] + azulBordes[14] + azulBordes[15] + azulBordes[16] + azulBordes[17] + 
                      azulBordes[18] + azulBordes[19] + azulBordes[20] + azulBordes[21] + azulBordes[22] + azulBordes[23] + azulBordes[24])//25)


            newImagSuavizado.putpixel((i,j),(abs(rojoFinalSuavizado),abs(verdeFinalSuavizado),abs(azulFinalSuavizado)))
            newImagBordes.putpixel((i,j),(abs(rojoFinalBordes),abs(verdeFinalBordes),abs(azulFinalBordes)))



#Se guarda la imagen resultante de la convolucion        
newImagSuavizado.save("leena512_Suavizado.bmp")
newImagBordes.save("leena512_Bordes.bmp")

#Se calcula la transformada de fourier de la imagen con el filtro de suavizado
#Y se grafica
plt.figure(2)
plt.xlabel("Ancho")
plt.ylabel("Largo")
plt.title("Transformada de filtro suavizado")
gray = newImagSuavizado.convert("L")
fourier= np.fft.fft2(gray)
fshift = np.fft.fftshift(fourier)
magnitudFFT= 20*np.log(np.abs(fshift))
plt.imshow(magnitudFFT)

#Se calcula la transformada de fourier de la imagen con el filtro de borde
#Y se calcula
plt.figure(3)
plt.xlabel("Ancho")
plt.ylabel("Largo")
plt.title("Transformada de filtro de borde")
gray = newImagBordes.convert("L")
fourier= np.fft.fft2(gray)
fshift = np.fft.fftshift(fourier)
magnitudFFT= 20*np.log(np.abs(fshift))
plt.imshow(magnitudFFT)


#Se muestra por pantalla ambas imagenes finales
newImagSuavizado.show()
newImagBordes.show()
plt.show()
