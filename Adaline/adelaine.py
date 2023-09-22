#Se creara la clase Adaline. Para ser utilizada luego en el adaline triple
import numpy as np

#Clase Adaline
class Adaline(object):
    def __init__(self, wi):
        #Variables a crear: 

        #Umbral (b)
        self.b = np.random.uniform(0,5)
        #T de aprendizaje (ratio)
        self.ratio = 0.01
        #Pesos (wi)
        self.wi = wi
        #Iteraciones itr
        self.itr = 200

    #Funcion de entrenanmiento: Recibe los datos para entrenar(d_X) y sus respectivas soluciones(d_y)
    def train(self, d_X, d_y):
        #Pasos a realizar:
        #Propagacion hacia delante e iteracion
        for i in range(self.itr): #Se repetira el entrenamiento por la cantidad de iteraciones que indiquemos
                #Calculamos las salidas (y) donde se realizara el producto punto entre los pesos (wi) y los datos de entrada (d_X)
                y = self.activacion(self.get_salida(d_X))
                #Calulamos la diferencia del error, El valor obtenido (y) menos Todos los valores esperados (d_y)
                error = (y - d_y) 
                #Por lo tanto los nuevos pesos seran los siguientes:
                self.wi += self.ratio * d_X.T.dot(error)
                #Y para el umbral: 
                self.b -= self.ratio * error.sum()
        return self

    
    def get_salida(self, X):
        return np.dot(X,self.wi) + self.b #Se utiliza para ajustar el punto de corte o sesgo del modelo
    
    def activacion(self, X): #Esta función solo retorna lo que se le envio
        return X
    
    def prediccion(self, test):
        #salida = []
        #for i in range(len(test)):
         #   salida.append()
        return np.where(self.activacion(self.get_salida(test)) >= 0.0, 1, -1) #salida
