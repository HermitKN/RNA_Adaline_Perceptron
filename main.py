import numpy as np
import matplotlib.pyplot as plano

from adelaine import Adaline

#Funcion donde creamos los datos de prueba 
def Crear_datos(num_inputs):

    aux = np.random.randint(0,43,size=(num_inputs,2))

    for i in range(0,int(len(aux))):
        if i >= len(aux)/2:
            aux[i] = np.random.randint(65,99,size=2)

    #Normalizar
    X_train = aux / 100

    y_train = np.zeros(num_inputs)

    for i in range(0,int(len(y_train))):
        if i >= len(y_train)/2:
            y_train[i] = 1

    return X_train, y_train


wi = np.random.uniform(0,1, size=2)
ad = Adaline(wi)

#Se normalizan los valores 
X_train, y_train = Crear_datos(250)

#Se envia a entrenar
ad.train(X_train,y_train)

#Se crean los datos de prueba
X_test, aux = Crear_datos(100)
datosx_g = X_test * 10
datosy_g = np.ones(100)

#Se prueban 
lista = ad.prediccion(X_test)
lista = np.asarray(lista)
datosy_g = lista

#Se imprimen
print(datosx_g/10)
print(datosy_g)
x = np.linspace(0, 10, 100)
y = -1*x + 10
plano.figure(figsize=(6,6))
plano.plot(x,y)
plano.scatter(datosx_g[datosy_g == -1].T[0],datosx_g[datosy_g == -1].T[1],marker="o", s=50, color = "red", linewidths=5, label= "Limpio")
plano.scatter(datosx_g[datosy_g == 1].T[0],datosx_g[datosy_g == 1].T[1],marker="o", s=50, color = "blue", linewidths=5, label= "Sucio")
plano.legend(bbox_to_anchor=(1.3,0.15))
plano.xlim(0,10)
plano.ylim(0,10)
plano.box(False)
plano.grid()
plano.show()