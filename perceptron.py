import matplotlib.pyplot as plano
import numpy as np

def Crear_datos(num_inputs):

    aux = np.random.randint(0,45,size=(num_inputs,2))

    for i in range(0,int(len(aux))):
        if i >= len(aux)/2:
            aux[i] = np.random.randint(55,99,size=2)

    #Normalizar
    X_train = aux / 100

    y_train = np.zeros(num_inputs)

    for i in range(0,int(len(aux))):
        if i >= len(aux)/2:
            y_train[i] = 1

    return X_train, y_train

casos, resultados  = Crear_datos(100)

#Funcion de Activación
def prediccion(pesos, x, b):
    z = x * pesos
    if z.sum() + b > 0:
        return 1
    else: 
        return 0

#Entrenamineto
pesos = np.random.uniform(0,2, size=2)
print("valor de pesos es:")
print(pesos)
base = np.random.uniform(0,2)
tasa_de_ap = 0.1
iteraciones = 200

for i in range (iteraciones):
    error_total = 0
    for j in range (len(casos)):
        calculo = prediccion(pesos,casos[j],base)
        error = resultados[j] - calculo
        error_total += error**2
        pesos[0] += tasa_de_ap * casos[j][0] * error
        pesos[1] += tasa_de_ap * casos[j][1] * error
        base += tasa_de_ap * error
    print(error_total, end=" ")
        
    
print("el valor del pesos es:")
print(pesos)
print("el valor del umbral es:")
print(base)
datos_de_pb, aux= Crear_datos(100)
respuestas = []

for cant in range (len(datos_de_pb)):
    respuestas.append(prediccion(pesos, datos_de_pb[cant] , base))

respuestas = np.array(respuestas)
print(str(datos_de_pb*100))
datos_de_pb = datos_de_pb * 10
x = np.linspace(0, 10, 100)
y = -1*x + 10
print('Las repuestas a cada dato: '+str(respuestas))
plano.figure(figsize=(6,6))
plano.plot(x,y)
plano.scatter(datos_de_pb[respuestas == 0].T[0],datos_de_pb[respuestas == 0].T[1],marker="o", s=50, color = "red", linewidths=5, label= "Limpio")
plano.scatter(datos_de_pb[respuestas == 1].T[0],datos_de_pb[respuestas == 1].T[1],marker="o", s=50, color = "blue", linewidths=5, label= "Sucio")
plano.xlim(0,10)
plano.ylim(0,10)
plano.box(False)
plano.grid()
plano.show()