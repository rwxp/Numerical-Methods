import matplotlib.pyplot as plt
import numpy as np
import math
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot

##Declaración de variables importantes:
h = 1
omegaX = 0.1 ## Termino de relajación.
## Dimensiones de la rejilla:
Nxmax=20
Nymax=20

## Dimensiones de las vigas:
anchoviga1 = 4
altoviga1 = 4
inicioviga1 = 8

anchoviga2 = 4
altoviga2 = 4
inicioviga2 = 16

##Dimensión matriz u:
dimensionU = Nxmax

## Declaracion de velocidad inicial, el agua se arrojara desde el costado izquierdo.
v0 = 1.0
tolerancia = 0.001

##PRUEBA CONVERGENCIA:

## Declaracion de la matriz de velocidades u.
u = np.zeros((Nxmax, Nymax), float)
## Declaracion terminos independientes bu.af
bu = np.zeros(Nxmax*Nymax)
##Inicializacion de terminos independientes con h/8.
for i in range(Nxmax*Nymax):
  bu[i]= h/8
## Declaracion de xu0 el cual es el vector que va a contener las soluciones a cada una de las ecuaciones resultantes del sistema.
xu0=np.zeros(Nxmax*Nymax)

def inicializa():
  for i in range(Nxmax):
    V = 0.3
    for j in range(Nymax):
      u[i, j] = round(V - 0.001, 3)
      V = V - 0.001

def inicializa_vigas(m):
  ##Viga 1:
  for i in range(Nymax-altoviga1, Nymax):
    for j in range(inicioviga1, inicioviga1+anchoviga1):
      u[i, j]=0
  ##Viga 2:
  for i in range(0, altoviga2):
    for j in range(inicioviga2, Nxmax):
      u[i, j]=0


## El siguiente algoritmo se encarga de generar la matriz jacobiana de U.
def gen_matriz_sis_lineal(n, u, h):
    a = 0
    b = 0
    c = 0
    d = 0
    matriz = []
    ## El tope maximo del bucle for es debido a que la matriz es de mxnXmxn
    for i in range(1, (n ** 2) + 1):
      fila = []
      for j in range(1, (n ** 2) + 1):
        ## ELIMINAR LA NO-LINEALIDAD: Si se busca eliminar la NO-LINEALIDAD entonces a y b valen -1/4.
        a = -1/4
        b = -1/4
        c = (h / 8) * u[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
        d = -(h / 8) * u[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4         

        # Primera diagonal
        ## Se encarga de inicializar la diagonal principal con 1s
        if (i == j):
          fila.append(1)

        # Segunda diagonal superior
        ## Se encarga de inicializar la segunda diagonal superior
        elif ((i % n != 0 and i + 1 == j)):
          fila.append(a)

        # Segunda diagonal inferior
        elif (
            ((i % n) - 1 != 0 and i == j + 1)):
          fila.append(b)

        # Tercera diagonal superior
        elif (i + n == j):
          fila.append(c)

        # Tercera diagonal inferior
        elif (i == j + n):
          fila.append(d)

        else:
          fila.append(0)
        ## Agrega la fila en la matriz
      matriz.append(fila)
    
    return matriz

##FUNCIONES QUE MODIFICAN LA JACOBIANA:
##La funcion surfaceG modifica la frontera de arriba.
def surfaceG(m,b):
  n=len(u)
  for i in range(0, n):
    b[i]=0
    for j in range(0, n*n):
      m[i][j] = 0
      if(j==i):
        m[i][j]= 1
## La funcion InletF modifica en la matriz Jacobiana la entrada al lado izquierdo.
def InletF(m, b):
  n=len(u)
  for i in range(0, n*n, n):
    b[i]=v0
    for j in range(0, n * n):
      m[i][j] = 0
      if i == j:
        m[i][j] = 1

## La funcion outlet se encarga 
def outlet(m, b):
    n = len(u)
    ## El bucle for externo se encarga de recorrer todas las ecuaciones de la derecha
    for i in range(n - 1, n * n, n):
      b[i] = 0
      ##El bucle for interno recorre todas las variables y las vuelve cero a excepción de la variable de esa fila, la cual vale 1.
      for j in range(0, n * n):
        m[i][j] = 0
        if i == j:
          m[i][j] = 1

def centerLine(m,b):
  n=len(u)
  for i in range(n*n-n,n*n):
    b[i] = 0
    for j in range(0,n*n):
      m[i][j] = 0
      if i == j:
        m[i][j] = 1
  return m,b

def transformPairToOne(i, j):
    return (i+1) * Nxmax - (Nxmax - j)

## Se encarga de llenar las vigas pero jacobiana.
def llenar_ecuaciones_vigas(m, b):
    n = len(u)
    # Rellenar
    # Viga 1
    for i in range(Nxmax - altoviga1, Nxmax):
        for k in range(inicioviga1, inicioviga1 + anchoviga1):
            for j in range(0, n * n):
                j2 = math.ceil(j / Nymax) - 1
                ## La funcion transformPairToOne se encarga de retornar el numero de fila (ecuacion) de los valores k,j
                m[transformPairToOne(i, k)][j] = 0
                if transformPairToOne(i, k) == j:
                    m[transformPairToOne(i, k)][j] = 1
    ## Agrega los terminos independientes para las ecuaciones de la viga. Estos terminos independientes valen cero.
    for i in range(Nymax - altoviga1, Nymax):
        for k in range(inicioviga1, inicioviga1 + anchoviga1):
            b[transformPairToOne(i, k)] = 0

    # Viga 2
    ## Se repite exactamente el proceso de la viga 1 pero con la viga 2.
    for i in range(0, altoviga2):
        for k in range(inicioviga2, inicioviga2 + anchoviga2):
            for j in range(0, n * n):
                j2 = math.ceil(j / Nymax) - 1
                m[transformPairToOne(i, k)][j] = 0
                if transformPairToOne(i, k) == j:
                    m[transformPairToOne(i, k)][j] = 1

    for i in range(0, altoviga2):
        for k in range(inicioviga2, inicioviga2 + anchoviga2):
            b[transformPairToOne(i, k)] = 0

def viga1(m, b):
  ##Sirve para definir los bordes de la viga1.
    n = len(u)
    # Pared izquierda viga 1
    ## Hace que los terminos independientes de las ecuaciones que estan al lado izquierdo de la viga1 sean cero
    for i in range(n - altoviga1, n):
      for j in range(0, n * n):
        b[i * n + inicioviga1 - 1] = 0

    # Pared superior viga 1
    ## Hace que los terminos independientes de las ecuaciones que estan en la parte superior de la viga1 sean ceros.
    f = n - altoviga1 - 1
    for j in range(inicioviga1, inicioviga1 + anchoviga1 + 1):
      b[transformPairToOne(f, j)] = 0
    
    # Pared derecha viga 1
    ##Hace que los terminos independientes de las ecuaciones que estan en la parte derecha de la viga1 sean ceros.
    for i in range(n - altoviga1 - 1, n):
        for j in range(0, n * n):
          b[i * n + inicioviga1 + anchoviga1] = 0

    ##------------------------------------------------
    # Pared izquierda viga 1
    ##Lo que hace es volver cero las variables en cada ecuacion de las casillas de izquierda de la viga, 
    ##a excepcion de la variable sobre la cual estoy.
    for i in range(n - altoviga1, n):
      for j in range(0, n * n):
        m[i * n + inicioviga1 - 1][j] = 0
        if i * n + inicioviga1 - 1 == j:
          m[i * n + inicioviga1 - 1][j] = 1
    # Pared superior viga 1
    f = n - altoviga1 - 1
    for j in range(inicioviga1, inicioviga1 + anchoviga1):
      for k in range(0, n * n):
        m[transformPairToOne(f, j)][k] = 0
        if transformPairToOne(f, j) == k:
          m[transformPairToOne(f, j)][k] = 1
    # Pared derecha viga 1
    for i in range(n - altoviga1 - 1, n):
      for j in range(0, n * n):
        m[i * n + inicioviga1 + anchoviga1][j] = 0
        if i * n + inicioviga1 + anchoviga1 == j:
          m[i * n + inicioviga1 + anchoviga1][j] = 1

def viga2(m, b):
    n = len(u)

    # Pared izquierda viga 2
    for i in range(0, altoviga2 + 1):
        for j in range(0, n * n):
          b[i * n + inicioviga2 - 1] = 0

    # Pared inferior viga 2
    f = altoviga2
    for j in range(inicioviga2, inicioviga2 + anchoviga2):
      b[transformPairToOne(f, j)] = 0

    # Pared izquierda viga 2
    for i in range(0, altoviga2 + 1):
      for j in range(0, n * n):
        m[i * n + inicioviga2 - 1][j] = 0
        if i * n + inicioviga2 - 1 == j:
          m[i * n + inicioviga2 - 1][j] = 1
        # Pared inferior viga 2
    f = altoviga2
    for j in range(inicioviga2, inicioviga2 + anchoviga2):
      for k in range(0, n * n):
        m[transformPairToOne(f, j)][k] = 0
        if transformPairToOne(f, j) == k:
          m[transformPairToOne(f, j)][k] = 1

def condiciones(mu, bu):
  surfaceG(mu, bu)
  InletF(mu, bu)
  outlet(mu, bu)
  centerLine(mu, bu)
  llenar_ecuaciones_vigas(mu, bu)
  viga1(mu, bu)
  viga2(mu, bu)



def muestra_matriz(m, name):
  print("La matriz ", name, " es la siguiente:")
  for row in m:
    for value in row:
      print("\t", round(value , 1), end=" ")
    print()

#inicializa()
inicializa_vigas(u)
jacU = gen_matriz_sis_lineal(dimensionU, u, h)
#muestra_matriz(jacU,"jacU")
condiciones(jacU, bu)

## APLICACIÓN DE LOS MÉTODOS DE SOLUCIÓN DEL SISTEMA.
# def Gauss_Seidel(A, x, b, N):
#   n=len(A)
#   k=0
#   while k < N:
#     suma=0
#     k+=1
#     for i in range(0, n):
#       suma=0
#       for j in range(0, n):
#         if(i!=j):
#           suma=suma+A[i][j]*x[j]
#       x[i]=(b[i]-suma)/A[i][i]


def Gauss_Seidel(A, x, b, N):
  tamano = np.shape(A)
  n = tamano[0]
  m = tamano[1]
  X = np.copy(x)
  diferencia = np.ones(n,dtype=float)
  errado = 2*tolerancia
  itera = 0
  while errado>tolerancia:
    for i in range(0,n,1):
      suma = 0
      for j in range(0,m,1):
        if (i!=j): 
          suma = suma-A[i][j]*X[j]
        
      nuevo = (b[i]+suma)/A[i][i]
      diferencia[i] = np.abs(nuevo-X[i])
      X[i] = nuevo
    errado = np.max(diferencia)
    itera = itera + 1

  print("CONVERGE en la iteración: ",itera)
  X = X = np.reshape(np.transpose([X]),(400))
  return X
 
def GS(A,x,b,N):
  L = np.tril(A)
  U = A-L
  for i in range(N):
    xi_1 = x
    x = np.dot(np.linalg.inv(L), b - np.dot(U,x))
  return x
  
# def Jacobi(A,x,b,N):
#   D = np.diag(A).reshape(x.shape) ##Extrae la diagonal de la matriz A. (25X1)
#   R = A - np.diagflat(D) ##Deja la matriz sin diagonal, es decir, con 0 en la diagonal
#   for i in range(N):
#     x = (b - np.dot(R,x))/D ##Genera el vector de soluciones utilizando el método de Jacobi
#   return x
def Jacobi(A,x,b, tolerancia):
  err=1
  D = np.diag(A).reshape(x.shape) ##Extrae la diagonal de la matriz A. (25X1)
  R = A - np.diagflat(D) ##Deja la matriz sin diagonal, es decir, con 0 en la diagonal
  i = 1
  while err>tolerancia:    
    x_old = np.copy(x)
    x = (b - np.dot(R,x))/D ##Genera el vector de soluciones utilizando el método de Jacobi
    err = np.absolute(np.linalg.norm(x) - np.linalg.norm(x_old))
    print("El error en la iteracion", i, "es", err)
    i+=1
  return x

def newtonRaphson(A,x,b,M,j):
  for i in range(0,M):
    ##deltax = Jacobi(A,x,b,j)
    deltax = Jacobi(A,x,b,j)
    x = x - deltax ##Toca que investigar si es mas o menos.
  return x

def NR1(A,x,b,M,tolerancia):
  for i in range(0,M):
    xold = np.copy(x)
    deltax = Jacobi(A,x,b,tolerancia)
    x = x - deltax ##Toca que investigar si es mas o menos.
    err = np.absolute(np.linalg.norm(x)-np.linalg.norm(xold))
    print("el error es",err)
    if(err<tolerancia):
      print("CONVERGE",i)
      break
  return x   
  

def NR2(A,x,b,tolerancia):
  err=1
  i = 1
  while err>tolerancia:
    x_old = np.copy(x)
    deltax = Jacobi(A,x,b,tolerancia)
    x = x - deltax
    err = np.absolute(np.linalg.norm(x) - np.linalg.norm(x_old))
    print("El error en la iteracion", i, "es", err)
    i+=1
  return x

def rs(A,x,b,N):
  for i in range(N):
    r=b-np.dot(A,x)
    x = x + r
  return x

def richardson(A,x,b, tolerancia):
  iteracion = 1
  err = 1
  while err>tolerancia:
    x_old = np.copy(x)
    r=b-np.dot(A,x)
    x = x + r
    err = np.absolute(np.linalg.norm(x) - np.linalg.norm(x_old))
    print("Es la iteracion", iteracion)
    iteracion+=1

  return x


for i in range(1):
    ## Envia la matriz u, el vector x que al inicio vale cero, los terminos independientes bu, 
    ##xu0 = newtonRaphson(jacU, xu0, -bu, 1, 20)
    ##xu0=rs(jacU, xu0, bu, 1)
    ##xu0=rs(jacU, xu0, bu, 1)
    xu0 = GS(jacU, xu0, bu, 1)
    ##xu0 = NR2(jacU, xu0, bu, tolerancia)
    it = 0
    for i in range(0, Nxmax):
        for j in range(0, Nymax):
            u[i, j] = u[i, j] + omegaX * xu0[it] ## Aparentemente es el vector de soluciones.
            it += 1
    newjacU = gen_matriz_sis_lineal(Nxmax, u, 1)
    condiciones(newjacU, bu)

##Muestra la matriz U solucion al final de todas las iteraciones:
muestra_matriz(u, "u")

############################################################
# Matriz de magnitudes

magn = np.zeros((Nxmax, Nxmax))

for j in range(Nxmax - 1):
    for i in range(Nxmax - 1):
        magn[i][j] = abs(u[i][j])
        # magn[i][j] = abs(u[i][j]+w[i][j])/2
        # magn[i][j] = [i][j]
# Mostrar(magn)
##############################################################
# Normalizamos las matrices halladas
def normalizar():
    for j in range(Nymax):
        for i in range(Nxmax):
            m = math.sqrt(u[i][j])
            if (m != 0):
                u[i][j] = u[i][j] / m

##normalizar()


##PREPARACION PARA MOSTRAR LA GRAFICA:
##Declara vectores que corresponderan a los ejes de la grafica.
x = np.linspace(0, Nxmax - 1, Nxmax)
y = np.linspace(0, Nymax - 1, Nymax)

xmesh, ymesh= np.meshgrid(x, y)
 
umesh = u

#####################################################
# Graficar
plt.imshow(magn)

plt.colorbar()
plt.quiver(xmesh, umesh)
plt.show()
##muestra_matriz(jacU, "jacobiana")
  


