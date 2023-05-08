import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
from matplotlib import pyplot
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1').colors

fig = plt.figure(figsize=(4,3))
ax = plt.axes(projection='3d')
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False


n_list=[2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100]
d_list=[100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000]
n_n=10  #10/18
n_d=10  # 10/24
# 3D surface
n=np.array(n_list[:n_n])
d=np.array(d_list[:n_d])
X, Y = np.meshgrid(n, d)
pd_file= pd.read_csv("./results/rst_mnist_acc_df.csv")
A=pd_file.values[:n_d,1:n_n+1]
ax.plot_surface(X,Y,A+1,cmap='rainbow')  # +1 to fall into values of the real accuracy
ax.set_xlabel(r'$N$', fontsize=12, rotation=50)
ax.set_ylabel(r'$D$', fontsize=12, rotation=-10)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Surrogate ensemble accuracy', rotation=90, fontsize=12)
plt.xticks(size= 12)
plt.yticks(size= 12)
ax.tick_params(axis='z',labelsize=12)
ax.legend(fontsize= 12)
ax.view_init(15,-150)
x_major_locator=MultipleLocator(5)
y_major_locator=MultipleLocator(300)
bx=plt.gca()
bx.xaxis.set_major_locator(x_major_locator)
bx.yaxis.set_major_locator(y_major_locator)
plt.tight_layout()
sfig=plt.gcf()
sfig.savefig('./results/mnist-df.pdf', format="pdf", bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(4,3))
ax = plt.axes(projection='3d')
pd_file= pd.read_csv("./results/rst_mnist_acc_real.csv")
B=pd_file.values[:n_d,1:n_n+1]
ax.plot_surface(X,Y,B,cmap='rainbow')
ax.set_xlabel(r'$N$', fontsize=12, rotation=50)
ax.set_ylabel(r'$D$', fontsize=12, rotation=-10)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('True ensemble accuracy', rotation=90, fontsize=12)
plt.xticks(size= 12)
plt.yticks(size= 12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(15,-150)
ax.legend(fontsize= 12)
x_major_locator=MultipleLocator(5)
y_major_locator=MultipleLocator(300)
bx=plt.gca()
bx.xaxis.set_major_locator(x_major_locator)
bx.yaxis.set_major_locator(y_major_locator)
plt.tight_layout()
sfig=plt.gcf()
sfig.savefig('./results/mnist-accu.pdf', format="pdf", bbox_inches='tight')
plt.show()

# Pearsonr
pccs = np.corrcoef((A).reshape(1,n_n*n_d),B.reshape(1,n_n*n_d))
print("Pearsonr coefficient:", pccs)

# 3D scatter
x=np.array([n_list[:n_n] for i in range(n_d)]).reshape(1,n_n*n_d)[0]
y=np.array([d_list[:n_d] for i in range(n_n)]).T.reshape(1,n_n*n_d)[0]
z=A.reshape(1,n_n*n_d)[0]
z1=B.reshape(1,n_n*n_d)[0]
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x, y, z, label='Evl')
# ax.set_xlabel('N', fontsize=12, rotation=50)
# ax.set_ylabel('D', fontsize=12, rotation=-10)
# ax.set_zlabel('Accuracy', fontsize=12)
# ax.view_init(15,-150)
# x_major_locator=MultipleLocator(4)
# bx=plt.gca()
# bx.xaxis.set_major_locator(x_major_locator)
# plt.show()

# # 3D fitting
# def func(data, a,b,c,d,e,f,g):
#     x,y = data
#     return (a*np.power(x,b)+c)*(d*np.log(y*e+f) + g)
def func(data, a,b,c,d,e,f,g,h,i):
    x,y = data
    return (a*np.log(x*b+c)+d)*(e*np.log(y*f+g) + h) +i

def func1(data, a,b,c,d,e,f,g,h,i):
    x,y = data
    return (a*np.log(x*b+c)+d)*(e*np.log(y*f+g) + h) +i +1


params, pcov = curve_fit(func, (x,y), z, maxfev=100000)
print(*params)
fig = plt.figure(figsize=(4,3))
ax = Axes3D(fig)
ax.scatter(x, y, z+1, c=palette[0], label='Evaluated')
ax.scatter(x, y, func1((x,y), *params), c=palette[1],label='Fitted' % params)
ax.legend(fontsize= 12)
ax.set_xlabel(r'$N$', fontsize=12, rotation=50)
ax.set_ylabel(r'$D$', fontsize=12, rotation=-10)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Surrogate ensemble accuracy', rotation=90, fontsize=12)
plt.xticks(size= 12)
plt.yticks(size= 12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(15,-150)
x_major_locator=MultipleLocator(5)
y_major_locator=MultipleLocator(300)
bx=plt.gca()
bx.xaxis.set_major_locator(x_major_locator)
bx.yaxis.set_major_locator(y_major_locator)
plt.tight_layout()
sfig=plt.gcf()
sfig.savefig('./results/mnist-df-fit.pdf', format="pdf", bbox_inches='tight')
plt.show()
# 0.5017119417210582 10.751182998179868 -21.418312530699435 -2.7778090816283028 -0.00890905445128486 5.73023367966792 -468.43367129201647 0.08896332243120453 0.0003351576270990361

params, pcov = curve_fit(func, (x,y), z1, maxfev=100000)
print(*params)
fig = plt.figure(figsize=(4,3))
ax = Axes3D(fig)
ax.scatter(x, y, z1, c=palette[0],label='Real')
ax.scatter(x, y, func((x,y), *params), c=palette[1],label='Fitted' % params)
ax.legend(fontsize= 12)
ax.set_xlabel(r'$N$', fontsize=12, rotation=50)
ax.set_ylabel(r'$D$', fontsize=12, rotation=-10)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('True ensemble accuracy', rotation=90, fontsize=12)
plt.xticks(size= 12)
plt.yticks(size= 12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(15,-150)
x_major_locator=MultipleLocator(5)
y_major_locator=MultipleLocator(300)
bx=plt.gca()
bx.xaxis.set_major_locator(x_major_locator)
bx.yaxis.set_major_locator(y_major_locator)
plt.tight_layout()
sfig=plt.gcf()
sfig.savefig('./results/mnist-accu-fit.pdf', format="pdf", bbox_inches='tight')
plt.show()
# 0.00030625506538928043 2.997047029160774 -5.866838062424441 9.29565543353712 0.002342501188356276 4.663997199764269 -352.23699101494634 17.021956681401335 -157.4690388764922

