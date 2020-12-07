# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:14:10 2020

@author: davii
"""

import pickle
from tkinter import *
from tkinter import messagebox 
import cv2 
import sys
import os
import imutils
import numpy as np
from PIL import Image
import pickle
import tkinter


def menu_pantalla():
    entrenandoRF()
    global ventana
    ventana= tkinter.Tk()
    ventana.geometry("400x420")
    ventana.title("Bienvenidos-UAM")
    ventana.iconbitmap("logo.ico")
    
    image=tkinter.PhotoImage(file="logos_uam.gif")
    image=image.subsample(2,2)
    tkinter.Label(ventana,image=image).pack()
    etiqueta = tkinter.Label(ventana, text="Acceso al sistema", bg="navy", fg="White", width="400", height="3", font=("calibri",15))
    etiqueta.pack() 
    tkinter.Label(ventana).pack()
    
    botonI = tkinter.Button(ventana, text="Reconocimiento Facial",bg="deepskyblue", fg="White", width="30", height="3", command=reconocimiento_facial)
    botonI.pack()
    botonR = tkinter.Button(ventana, text="Registar Usuario",bg="deepskyblue", fg="White", width="30", height="3", command=registro)
    botonR.pack()
    ventana.mainloop()
  

def registro():
    global pantalla2
    pantalla2 = tkinter.Toplevel(ventana)
    pantalla2.geometry("400x420")
    pantalla2.title("Registro-UAM")
    pantalla2.iconbitmap("logo.ico")
    etiqueta = tkinter.Label(pantalla2, text="REGISTRO", bg="deeppink", fg="White", width="400", height="3", font=("calibri",15))
    etiqueta.pack() 
    tkinter.Label(ventana,text=" " ).pack()
    
    global name      
    name= tkinter.StringVar()
    
    etiqueta = tkinter.Label(pantalla2, text="Ingrese su Nombre", fg="deeppink", width="400", height="2", font=("calibri",15)).pack()
    tkinter.Entry(pantalla2, textvariable=name).pack()
    tkinter.Label(pantalla2).pack()
    botonR = tkinter.Button(pantalla2, text="Registar",bg="deeppink", fg="White", width="30", height="3", command=capturandoRostros)
    botonR.pack()


def capturandoRostros():
    pantalla2.destroy()
    personName = name.get()
    dataPath = "D:/Downloads/prueba_reco/prueba_reco/data" #Cambia a la ruta donde hayas almacenado Data
    personPath = dataPath + '/' + personName
    
    if not os.path.exists(personPath):
    	print('Carpeta creada: ',personPath)
    	os.makedirs(personPath)
    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0
    
    while True:
    
    	ret, frame = cap.read()
    	if ret == False: break
    	frame =  imutils.resize(frame)
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	auxFrame = frame.copy()
    
    	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    	for (x,y,w,h) in faces:
    		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    		rostro = auxFrame[y:y+h,x:x+w]
    		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
    		cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
    		count = count + 1
    	cv2.imshow('frame',frame)
    
    	k =  cv2.waitKey(1)
    	if k == 27 or count >= 300:
    		break
    
    entrenandoRF()
    cap.release()
    cv2.destroyAllWindows()
    

def entrenandoRF():
    dataPath = 'D:/Downloads/prueba_reco/prueba_reco/data' #Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)
    
    labels = []
    facesData = []
    label = 0
    
    for nameDir in peopleList:
    	personPath = dataPath + '/' + nameDir
    	print('Leyendo las imágenes')
    
    	for fileName in os.listdir(personPath):
    		print('Rostros: ', nameDir + '/' + fileName)
    		labels.append(label)
    		facesData.append(cv2.imread(personPath+'/'+fileName,0))
    		image = cv2.imread(personPath+'/'+fileName,0)
            
            
    	label = label + 1
        
    # Métodos para entrenar el reconocedor
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    
    # Almacenando el modelo obtenido
    face_recognizer.write('modeloEigenFace.xml')
    print("Modelo almacenado...")
    
def NuevoUsuario():
    global ventana1
    ventana1= tkinter.Tk()
    ventana1.geometry("400x420")
    ventana1.title("Bienvenidos-UAM")
    ventana1.iconbitmap("logo.ico")
    
    image=tkinter.PhotoImage(file="logos_uam.gif")
    image=image.subsample(2,2)
    tkinter.Label(ventana,image=image).pack()
    etiqueta = tkinter.Label(ventana1, text="Acceso al sistema", bg="navy", fg="White", width="400", height="3", font=("calibri",15))
    etiqueta.pack() 
    tkinter.Label(ventana1).pack()
    
    botonR = tkinter.Button(ventana1, text="Registar Usuario",bg="deepskyblue", fg="White", width="30", height="3", command=registro)
    botonR.pack()
    ventana1.mainloop()
def reconocimiento_facial():
    dataPath = "D:/Downloads/prueba_reco/prueba_reco/data" #Cambia a la ruta donde hayas almacenado Data
    imagePaths = os.listdir(dataPath)
    print('imagePaths=',imagePaths)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Leyendo el modelo
    face_recognizer.read('modeloEigenFace.xml')
    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture('Imagenes y Videos de Prueba\\juan2.mp4')
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    while True:
        ret,frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            print(result)
            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            
            if result[1] < 70:
                cv2.putText(frame,'{}'.format("Bienvenido" + imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                NuevoUsuario()
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    
    
menu_pantalla()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







