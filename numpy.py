#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 


# In[4]:


# escalares 
escalar = np.array(5)
print(escalar)


# In[6]:


#vector 
vector = np.array([1,2,3,4,5,6])
print(vector)


# In[9]:


# matrices con llenado 
matriz = np.arange(9).reshape(3,3)
print(matriz)


# In[11]:


#tensores 
tensor = np.arange(12).reshape(3,2,2)
print(tensor)


# In[12]:


# objeto 
np.arange(6).reshape(3,2)


# In[13]:


# como utilizar el np.arange 
np.arange(start = 2, stop = 10, step = 2)


# In[14]:


#tambien sirven con indices negativos 
np.arange(11,1,-2)


# ## numpy.linspace()
# 
# se utiliza para generar un arreglo de valores igualmente **espaciados** dentro de un rango específico.
# 
# **numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)**
# 
# 
# ### funciones 
# 
# - **start**: El valor inicial del arreglo.
# - **stop:** El valor final del arreglo.
# - **num:** El número de elementos que se desean generar (por defecto es 50).
# - **endpoint:** Si es True, el valor final se incluirá en el arreglo. Si es False, el valor final no se incluirá.
# - **retstep:** Si es True, la función también devuelve el tamaño del paso entre los valores.
# - **dtype:** El tipo de datos de los elementos en el arreglo (opcional).
# 
# 
# 

# In[17]:


#arrelo de valores igualmente espaciados 

np.linspace(start = 11, stop = 12, num = 20)


# # Matplotlib
# - se hace para entender un poco la geometria de los datos 

# In[25]:


import matplotlib.pyplot as plt
linea = np.linspace(0, 1 , num = 10)
linea 
plt.plot(linea, '*')


# In[26]:


# vector con ceros y unos
print(np.zeros(4))
print(np.ones(6))


# In[29]:


#matriz ceros y unos 
print(np.zeros((2,2)))
print(np.ones((3,3)))


# 
# La función **numpy.full()** se utiliza para crear un arreglo NumPy con un tamaño y forma específicos,
# lleno de un valor constante. Es decir, llena todas las posiciones del arreglo con el mismo valor que 
# se especifica como argumento.
# La sintaxis básica de numpy.full() es la siguiente:
#     **numpy.full(shape, fill_value, dtype=None)**
#     
# Donde:
# 
# - shape: La forma (dimensiones) del arreglo que deseas crear, especificada como una tupla de enteros.
# - fill_value: El valor constante que deseas llenar en todas las posiciones del arreglo.
# - dtype: El tipo de datos de los elementos en el arreglo (opcional).
# - Aquí tienes un ejemplo de cómo usar numpy.full():

# In[30]:


print(np.full(shape = (2,2), fill_value = 5))


# In[31]:


print(np.full(shape = (2,3,4), fill_value = 0.55))


# In[32]:


help(np.linspace)


# La función **numpy.random.rand()** se utiliza para generar números aleatorios en un arreglo NumPy con valores distribuidos uniformemente en el intervalo [0, 1). Es decir, crea un arreglo de números aleatorios en el rango de 0 (incluido) a 1 (excluido), utilizando una distribución uniforme.
# 
# La sintaxis básica de numpy.random.rand() es la siguiente:

# - numpy.random.rand(d0, d1, ..., dn)
# 

# In[41]:


# llenar un array con numeros aleatorios
randoms = (np.random.rand(100))
randoms


# La función **plt.hist()** es parte del módulo matplotlib.pyplot en Python y se utiliza para crear y visualizar un histograma a partir de un conjunto de datos. Un histograma es una representación gráfica de la distribución de frecuencias de los datos en intervalos (bins) específicos. Es útil para comprender la distribución de un conjunto de datos y detectar patrones o tendencias.
# 
# La sintaxis básica de plt.hist() es la siguiente:
# **plt.hist(x, bins=None, range=None, density=False, cumulative=False, ...)**
# Donde:
# 
# - x: Los datos que deseas visualizar en el histograma.
# - bins: El número de intervalos (bins) en los que se dividirán los datos. También puedes proporcionar una lista de valores que representan los límites de los bins.
# - range: El rango de valores que se incluirán en el histograma. Puedes especificar un par de valores para establecer los límites.
# - density: Si es True, el histograma representará una densidad de probabilidad en lugar de frecuencias.
# - cumulative: Si es True, el histograma mostrará la distribución acumulativa.

# In[43]:


# grafica de histograma a partir del array generado con numeros aleatorios del 0 a 1 plt.figure("random")
n, bins, patches = plt.hist(randoms, bins=100, facecolor='green', alpha=1)  
plt.show()


# # A continuacion todas las distribuccion de numeros aleatorios 

# In[45]:


Distribuccion_unifrome = np.random.rand(100)
##
n, bins, patches = plt.hist(Distribuccion_unifrome, bins=100, facecolor='green', alpha=1)  
plt.show()


# In[50]:


numeros_aleatorios = np.random.random(100)
print(numeros_aleatorios)
n, bins, patches = plt.hist(numeros_aleatorios, bins=100, facecolor='green', alpha=1)  
plt.show()


# In[54]:


# distribuccion gaussiana (media 0, desviación estándar 1).
distribuccion_normal_estandar = np.random.randn(100)
print(distribuccion_normal_estandar)
n, bins, patches = plt.hist(distribuccion_normal_estandar, bins=100, facecolor='green', alpha=1)  
plt.show()


# In[56]:


# distribuccion gaussiana (media 0, desviación estándar 1).
distribuccion_normal = np.random.normal(100)
print(distribuccion_normal)
n, bins, patches = plt.hist(distribuccion_normal, bins=10, facecolor='green', alpha=1)  
plt.show()


# demas distribucciones 
# Generadores de Números Aleatorios de Distribución Exponencial:
# 
# - numpy.random.exponential(): Genera números aleatorios con una distribución exponencial con una tasa especificada.
# Generadores de Números Aleatorios de Distribución Poisson:
# 
# - numpy.random.poisson(): Genera números aleatorios con una distribución de Poisson con una tasa especificada.
# Generadores de Números Aleatorios de Distribución Binomial:
# 
# - numpy.random.binomial(): Genera números aleatorios con una distribución binomial con parámetros de tamaño y probabilidad de éxito.
# Generadores de Números Aleatorios de Distribución Uniforme Discreta:
# 
# - numpy.random.choice(): Genera muestras aleatorias de un conjunto de valores.

# In[57]:


# diferencias con las listas en python 
#operaciones 
A = np.arange(5,11)
A


# In[58]:


A + 10


# In[59]:


## Operaciones con listas ---> Diferencias con los arrays de Numpy
a = [1,2,3,4]
b = [4,5,8,14]
def restlistas(a,b):
    resta=[]
    for i in range(len(a)):
        resta.append(a[i]-b[i])
    return resta


# In[62]:


## Operaciones con listas ---> Diferencias con los arrays de Numpy
a = [1,2,3,4]
b = [4,5,8,14,15]
def restlistas(a,b):
    resta=[]
    for i in range(len(a)):
        resta.append(a[i]-b[i])
    return resta


# In[63]:


restlistas(a,b)


# In[64]:


B = np.arange(0,6)


# In[65]:


A - B


# In[67]:


# diferencias de tiempo
time
for i in range(100000):
    print(i)


# In[70]:


import time

# Guardar el tiempo de inicio
inicio = time.time()

# Tu código o porción de código aquí
for i in range(10000):
    print(i)

# Guardar el tiempo de finalización
fin = time.time()

# Calcular el tiempo transcurrido
tiempo_transcurrido = fin - inicio

print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")


# In[71]:


import time

# Guardar el tiempo de inicio
inicio = time.time()

# Tu código o porción de código aquí
np.arange(10000)

# Guardar el tiempo de finalización
fin = time.time()

# Calcular el tiempo transcurrido
tiempo_transcurrido = fin - inicio

print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")


# In[ ]:




