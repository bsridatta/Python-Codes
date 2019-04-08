#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
import numpy , random , math
from scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
v=0.2
classA = numpy. concatenate((numpy.random.randn(10, 2) * v + [1.5, 0.5],
numpy.random.randn(10, 2) * v + [-1.5 ,0.5]))
classB = numpy.random.randn(20, 2) * v + [3.0, 1.5]
inputs = numpy.concatenate (( classA , classB ))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]) , -numpy.ones(classB.shape[0])))
N =inputs.shape[0]
start=numpy.zeros(N)
C=10
numpy.random.seed(100)
permute=list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]

# ### Kernel function

# In[9]:


def linear_kernel(x,y):
    return numpy.dot(numpy.transpose(x),y)

def poly_kernel(x,y,p):
    return numpy.power(numpy.dot(numpy.transpose(x),y) + 1,p)

def rbf_kernel(x,y,sigma):
    #distance=numpy.sqrt(x**2 - y**2)
    distance = numpy.linalg.norm(x-y)

    result=numpy.power(distance,2)
    smooth=2*numpy.power(sigma,2)
    return numpy.exp(-(result/smooth))


# ### Objective

# In[10]:


def objective(alpha):
    #dualScalar=1/2*numpy.sum(numpy.dot(alpha,P))-numpy.sum(alpha)
    dualScalar=0
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            dualScalar+=alpha[i]*alpha[j]*P[i][j]

    sumAlpha=0
    for i in range(len(inputs)):
        sumAlpha+=alpha[i]

    dualScalar=1/2*dualScalar-sumAlpha

    return dualScalar


# ### Zero Fun

# In[11]:


xZ=0


# In[12]:


def zerofun(alpha):
    xZ= numpy.dot(alpha,targets)
    return xZ


# ### Data and initial params

# In[13]:





# In[14]:




# In[15]:




# ### Gobal Matrix P

# In[16]:


P=numpy.zeros((len(inputs),len(inputs)))
for i in range(len(inputs)):
    for j in range(len(inputs)):
        P[i][j]=targets[i]*targets[j]*rbf_kernel(inputs[i][0:2],inputs[j][0:2],5)


#  ### Optimization function

# In[17]:



ret = minimize(objective,start, bounds=[(0, C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun})
alpha = ret['x']

# alpha=start
# for i in range(100):
#     print i
#     ret = minimize(objective,alpha, bounds=[(0, None) for b in range(N)], constraints={'type':'eq', 'fun':zerofun})
#     alpha = ret['x']



# ### Extract non zero Alphas

# In[18]:



threshold=10e-5
support=[]
for i in range(len(alpha)):
    if(threshold < alpha[i]):
        support.append([alpha[i],inputs[i], targets[i]])


# for x in support:



# ### B

# In[19]:


bias=0
for i in range(len(inputs)):
    if(alpha[i]>0 and alpha[i]<C):
        pK=rbf_kernel(support[0][1],inputs[i],5)
        bias=bias+alpha[i]*targets[i]*pK

bias-=support[1][2]
print(bias)



# # The indicator function
# Implement the indicator function (equation 6) which uses the non-zero
# α i ’s together with their ⃗x i ’s and t i ’s to classify new points.
#
#
# def poly_kernel(x,y,p):
#     return numpy.power(numpy.dot(numpy.transpose(x),y) + 1,p)
#

# In[20]:


def indicator(x,y):
    new=[x,y]
    pred=0

    for i in range(len(inputs)):
        if(alpha[i]>0 and alpha[i]<C):
            pK=rbf_kernel(new,inputs[i],5)
            pred+=alpha[i]*targets[i]*pK

    pred-=bias
    #print(pred)
    return pred



# # Test Data

# In[21]:





# # Plotting

# In[22]:


plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.')
plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.')
xgrid=numpy.linspace(-5, 5)
ygrid=numpy.linspace(-4, 4)
grid=numpy.array([[indicator(x,y)
                   for x in xgrid ]
                  for y in ygrid])
plt.contour(xgrid,ygrid,grid,
               (-1.0, 0.0, 1.0),
               colors=('red', 'black', 'blue'),
               linewidths=(1, 3, 1))
plt.axis('equal')
plt.savefig('svm.pdf')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
