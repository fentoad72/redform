#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

os.chdir('/Users/arbetter/Desktop/python/RedForm/')

sys.path.append(os.getcwd())

import redform_mod as rfm

# test guass-jordan with 3x3 matrix

def gj_driver():

    Atrue = np.array([[1,1,1],[0,2,5],[2,5,-1]],dtype=np.longdouble)
    Bvect = np.array([6,-4,27],dtype=np.longdouble)

    Ainv, Bsolv = gaussjordan(Atrue,Bvect)

    Pmod = np.zeros(len(Bvect))
    dPdx = np.zeros([len(Bvect),len(Bvect)],dtype=np.longdouble)

    for i in range(len(Bsolv)):
        Pmod[i],dPdx[i][:] = rfm.poly_func(Atrue[i][:],Bsolv[i])
#        print 'i',i,Bsolv[i],Pmod[i],dPdx[i][:]

    #Xvect = np.copy(Bsolv)
    #Xvect = np.append(Xvect,(-1,-5,2,9))

    #Atrue = np.reshape(np.tile(Atrue,9),(9,9))
    del Pmod, dPdx, Ainv, Atrue, Bvect

    return


### MRQ


# error flag -- will become true if grand_func is called with an unrecognized form
error = False

tiny = np.longdouble(1.e-6)  # small number to prevent divide by zero
tiny_sq = tiny*tiny

#Xvect = np.asarray([0.2,-1.3,4.4,-5.8,7.6,-9.0,9.9,-2.8],dtype=np.longdouble)
#Xvect = np.arange(51,dtype=np.longdouble)*4/10. - 10.

#Xextend = np.arange(21,dtype=np.longdouble) + 10.

#Xvect = np.append(Xvect,Xextend)

#Xvect = np.asarray([-1.,0.,1.],dtype=np.longdouble)





#Xvect = np.arange(16,dtype=np.longdouble)/5.- 3.

# for a vector Xvect with N data points x(N) and solutions Y(N), stddev Yerr(N)
# M coefficients in A (so A is a M x M array)

### Assume we have same number of obs for each field Xvect
#Yerr = np.zeros(len(Xvect),dtype=np.longdouble)

# Definte Actual function (Y-hat) as sum of a polynomial and 2 exponentials

Bpoly = np.asarray([1., 0.1 ,0.001],dtype=np.longdouble)
#Bpoly = [0.,0.,0.]

exp_factor = np.asarray([10.,-5.,5.],dtype=np.longdouble)

Bgauss = np.asarray([20.,5.5,2.5],dtype=np.longdouble)
#Bgauss = [0.,0.,0.]

Xtrue = np.arange(501,dtype=np.longdouble)/10. - 10.

x1 = -10
x2 = 15
y1 = -10
y2 = 60

Ytrue = np.zeros(len(Xtrue),dtype=np.longdouble)  # aka Y-hat
Ypoly = np.zeros(len(Xtrue),dtype=np.longdouble)
Yexp = np.zeros(len(Xtrue),dtype=np.longdouble)
Ygauss = np.zeros(len(Xtrue),dtype=np.longdouble)
Ycos = np.zeros(len(Xtrue),dtype=np.longdouble)

### Pick 3 unique Xvects...

Xvect1 = np.asarray([-7.,-2.,1.,4.,8.,11.,13.,],dtype=np.longdouble)
Xvect2 = np.asarray([-9.,-8.,-5.,-3.,-1.,15.,7.],dtype=np.longdouble)
Xvect3 = np.asarray([0.,2.,3.,5.,6.,12.,14.],dtype=np.longdouble)

### and 3 Y functions correspond

Ypoly_obs = np.zeros(len(Xvect1),dtype=np.longdouble)
Yexp_obs = np.zeros(len(Xvect2),dtype=np.longdouble)
Ygauss_obs = np.zeros(len(Xvect3),dtype=np.longdouble)
Ycos_error1 = np.zeros(len(Xvect1),dtype=np.longdouble)
Ycos_error2 = np.zeros(len(Xvect2),dtype=np.longdouble)
Ycos_error3 = np.zeros(len(Xvect3),dtype=np.longdouble)

Y_act1 = np.zeros(len(Xvect1),dtype=np.longdouble)
Y_act2 = np.zeros(len(Xvect2),dtype=np.longdouble)
Y_act3 = np.zeros(len(Xvect3),dtype=np.longdouble)

### get the true values which are y = y(A1,x1) + y(A2,x2) + y(A3,x3)
#A1 = polynomial
#A2 = exponential
#A3 = exponential (guass)

for i in range(0,len(Xtrue)):

    Ypoly[i],dum1 = rfm.poly_func(Bpoly,Xtrue[i])
#    Ypoly[i] = 0.
    Yexp[i], dum2 = rfm.exp_func(exp_factor,Xtrue[i])
#    Yexp[i] = 0.
    Ygauss[i],dum3 = rfm.gauss_func(Bgauss,Xtrue[i])
    Ycos[i] =  0.5 *np.cos((Xtrue[i]*np.pi/np.sqrt(7.)))
    Ytrue[i] = Ytrue[i]  + Yexp[i]+ Ypoly[i] + Ygauss[i]
    print i, Xtrue[i],Ytrue[i]

del i

# get measurements which vary off of Y-hat by a cosine
for k in range(len(Xvect1)):
    Ypoly_obs[k],dum1 = rfm.poly_func(Bpoly,Xvect1[k])
    Ycos_error1[k] = 0.5 * np.cos((Xvect1[k]*np.pi/np.sqrt(7.)))
    Ypoly_obs[k] += Ycos_error1[k]
    Y1,dum1 = rfm.poly_func(Bpoly,Xvect1[k])
    Y2,dum2 = rfm.exp_func(exp_factor,Xvect1[k])
    Y3,dum3 = rfm.gauss_func(Bgauss,Xvect1[k])
    Y_act1[k] = Y1 + Y2 + Y3

    Yexp_obs[k], dum2 = rfm.exp_func(exp_factor,Xvect2[k])
    Ycos_error2[k] =  0.5 * np.cos((Xvect2[k]*np.pi/np.sqrt(7.)))
    Yexp_obs[k] += Ycos_error2[k]
    Y1,dum1 = rfm.poly_func(Bpoly,Xvect2[k])
    Y2,dum2 = rfm.exp_func(exp_factor,Xvect2[k])
    Y3,dum3 = rfm.gauss_func(Bgauss,Xvect2[k])
    Y_act2[k] = Y1 + Y2 + Y3

    Ygauss_obs[k],dum3 = rfm.gauss_func(Bgauss,Xvect3[k])
    Ycos_error3[k] = 0.5 * np.cos((Xvect3[k]*np.pi/np.sqrt(7.)))
    Ygauss_obs[k] += Ycos_error3[k]
    Y1,dum1 = rfm.poly_func(Bpoly,Xvect3[k])
    Y2,dum2 = rfm.exp_func(exp_factor,Xvect3[k])
    Y3,dum3 = rfm.gauss_func(Bgauss,Xvect3[k])
    Y_act3[k] = Y1 + Y2 + Y3

del k,Y1,Y2,Y3
del Bpoly,Bgauss,exp_factor

fig1 = plt.figure()
plt.plot(Xtrue,Ytrue,'black')
plt.plot(Xtrue,Ypoly,'blue')
plt.plot(Xtrue,Yexp,'red')
plt.plot(Xtrue,Ygauss,'green')
plt.plot(Xvect1,Y_act1+Ycos_error1,'bo')
plt.plot(Xvect2,Y_act2+Ycos_error2,'go')
plt.plot(Xvect3,Y_act3+Ycos_error3,'ro')
#plt.plot(Xvect,Yvect,'ko')


plt.title('Input Components soln=black\nblue=poly green=gauss red=expnt')
plt.axis([x1, x2, y1, y2])
plt.show()
plt.savefig('redform_inputs.png')

del Ypoly,Yexp, Ycos,dum1,dum2,dum3

### Same Xvect & Yvect for all attempts

### First try: one polynomial

# Specify # fitting forms
nfunc = 1
nexp = 0
npoly = nfunc - nexp


FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n,3*n+1,3*n+3] = 1.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 0.

Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.

del n,k,nfunc,npoly,nexp

# since poly_func and exp_func have 3 coefficients, FuncFit

Xvect = np.append(np.append(Xvect1,Xvect2),Xvect3)
# save copy of Xvect for later
Xsave = Xvect

Yvect = np.append(np.append(Y_act1+Ycos_error1,Y_act2+Ycos_error2),Y_act3+Ycos_error1)

Yerr = np.zeros(len(Yvect),dtype=np.longdouble)
Yerr[:] = np.std(Yvect,dtype=np.longdouble)

del Y_act1,Y_act2,Y_act3
del Ycos_error1,Ycos_error2,Ycos_error3

error = False

Asolv1, chi_sq1,covar1, alpha1 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)

#for j in range(len(Xvect)):
#    Emod[j],dEdx[j] = exp_func(Amatrix[j][:],Xvect[j])
#    print 'j',j,Xvect,Emod[i],dEdx[i]

### Yfunc1 is the function created by the fit form

Yfunc1 = np.zeros(len(Xtrue),dtype=np.longdouble)
dydx1 = np.zeros([len(Xtrue),len(Avector)],dtype=np.longdouble)
Ysum=0.
## need to count from 1 to N (all Xvect points)
Yfunc1,dydx1 = rfm.grand_func(Asolv1,Xtrue,FuncFit,error)

### let's call Ypoints the points prediced for Xvect
Ypoints1 = np.zeros(len(Xvect),dtype=np.longdouble)
dydum1 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ypoints1,dydum1 = rfm.grand_func(Asolv1,Xvect,FuncFit,error)

pass

fig2 = plt.figure()
#plt.plot(Xtrue,Ytrue, color='black')
#plt.axis([Xtrue[0],Xtrue[len(Xtrue)-1],-1500.,1500.])

plt.plot(Xvect,Yvect,'bo')
plt.plot(Xtrue,Yfunc1,color='blue')


            ### Second try: one gaussian

nfunc = 1
nexp = 1
npoly = nfunc - nexp

FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n:3*n+3] = 2.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 0.

Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.

del n,k,nfunc,npoly,nexp


#Xvect = np.copy(Xvect2)
#Yvect = np.copy(Y_act2+Ycos_error2)

#Yerr = np.zeros(len(Yvect),dtype=np.longdouble)
#Yerr[:] = np.std(Yvect,dtype=np.longdouble)

Asolv2, chi_sq2,covar2,alpha2 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)
#Asolv2 = np.zeros(len(Avector))

Yfunc2 = np.zeros(len(Xtrue),dtype=np.longdouble)
dydx2 = np.zeros([len(Xtrue),len(Avector)],dtype=np.longdouble)

## need to count from 1 to N (all Xvect points)

Yfunc2, dydx2 = rfm.grand_func(Asolv2,Xtrue,FuncFit,error)

Ypoints2 = np.zeros(len(Xvect),dtype=np.longdouble)
dydum2 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ypoints2, dydum = rfm.grand_func(Asolv2,Xvect,FuncFit,error)

plt.plot(Xvect,Yvect,'go')
plt.plot(Xtrue,Yfunc2,color='green')

#plt.show()

            ### Third try: exponential

nfunc = 1
nexp = 1
npoly = nfunc - nexp

FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n:3*n+3] = 1.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 0.

Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.


del n,k,nfunc,npoly,nexp

#Xvect = np.copy(Xvect3)
#Yvect = np.copy(Y_act3+Ycos_error3)

#Yerr = np.zeros(len(Yvect),dtype=np.longdouble)
#Yerr[:] = np.std(Yvect,dtype=np.longdouble)

Asolv3, chi_sq3,covar3,alpha3 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)
#Asolv2 = np.zeros(len(Avector))

Yfunc3 = np.zeros(len(Xtrue),dtype=np.longdouble)
dydx3 = np.zeros([len(Xtrue),len(Avector)],dtype=np.longdouble)

Yfunc3, dydx3 = rfm.grand_func(Asolv3,Xtrue,FuncFit,error)

Ypoints3 = np.zeros(len(Xvect),dtype=np.longdouble)
dydum3 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ypoints3, dydum3 = rfm.grand_func(Asolv3,Xvect,FuncFit,error)


plt.plot(Xvect,Yvect,'ro')
plt.plot(Xtrue,Yfunc3,color='red')

plt.plot(Xtrue,Ytrue,color='black')

plt.suptitle('Solutions 1 var (black=actual):\nblue=poly green=gauss red=expnt')
plt.axis([x1,x2,y1,y2])
plt.show()

plt.savefig('redform_fit1var.png')



            ### Fourth try: one poly, one gaussian

nfunc = 2
nexp = 1
npoly = nfunc - nexp

FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n:3*n+3] = 2.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 0.


Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.

del n,k,nfunc,npoly,nexp

# double length Xvect
Xvect = np.append(Xsave,Xsave)
# and double length Xtrue
XdoubleTrue = np.append(Xtrue,Xtrue)

Asolv4, chi_sq4,covar4,alpha4 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)
#Asolv2 = np.zeros(len(Avector))

Yfunc4 = np.zeros(len(Xtrue),dtype=np.longdouble)
dydx4 = np.zeros([len(XdoubleTrue),len(Avector)],dtype=np.longdouble)

Yfunc4,dydx4 = rfm.grand_func(Asolv4,XdoubleTrue,FuncFit,error)

Ypoints4 = np.zeros(len(Xsave),dtype=np.longdouble)
dydum4 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ysum = 0

Ypoints4,dydum4 = rfm.grand_func(Asolv4,Xvect,FuncFit,error)

fig3=plt.figure()
plt.plot(Xsave,Yvect,'bo')
plt.plot(Xtrue,Yfunc4,'blue')


            ### Fifth try: one poly, one expt

nfunc = 2
nexp = 1
npoly = nfunc - nexp

FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n:3*n+3] = 1.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 0.


Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.

del n,k,nfunc,npoly,nexp

# Xvect is from before, double length

#Yerr = np.zeros(len(Yvect),dtype=np.longdouble)
#Yerr[:] = np.std(Yvect,dtype=np.longdouble)

Asolv5, chi_sq5,covar5,alpha5 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)
#Asolv2 = np.zeros(len(Avector))

Yfunc5 = np.zeros(len(Xtrue),dtype=np.longdouble)
dydx5 = np.zeros([len(XdoubleTrue),len(Avector)],dtype=np.longdouble)

Yfunc5,dydx5 = rfm.grand_func(Asolv5,XdoubleTrue,FuncFit,error)

Ypoints5 = np.zeros(len(Xsave),dtype=np.longdouble)
dydum5 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ypoints5,dydum5 = rfm.grand_func(Asolv5,Xvect,FuncFit,error)

plt.plot(Xsave,Yvect,'ro')
plt.plot(Xtrue,Yfunc5,'red')


            ### Sixth try: expn't and guassian

nfunc = 2
nexp = 2
npoly = nfunc - nexp

FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n:3*n+3] = 2.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 1.

Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.

del n,k,nfunc,npoly,nexp

#Xvect is double length from above

Asolv6, chi_sq6,covar6,alpha6 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)
#Asolv2 = np.zeros(len(Avector))

Yfunc6 = np.zeros(len(Xtrue),dtype=np.longdouble)
dydx6 = np.zeros([len(XdoubleTrue),len(Avector)],dtype=np.longdouble)

Yfunc6,dydx6 = rfm.grand_func(Asolv6,XdoubleTrue,FuncFit,error)

Ypoints6 = np.zeros(len(Xsave),dtype=np.longdouble)
dydum6 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ypoints6, dydum6 = rfm.grand_func(Asolv6,Xvect,FuncFit,error)


plt.plot(Xsave,Yvect,'go')
plt.plot(Xtrue,Yfunc6,'green')

plt.plot(Xtrue,Ytrue,'black')


#plt.suptitle('Fitting Forms')
plt.suptitle('Solutions 2 var (black=actual):\nblue=poly+gauss green=expt+gauss red=poly+expt')
plt.axis([x1,x2,y1,y2])
plt.show()

plt.savefig('fit_2var.png')

            ### Seventh try: both gaussians

nfunc = 3
nexp = 2
npoly = nfunc - nexp

FuncFit = np.zeros(nfunc*3,dtype=long)

for n in range(0,nexp):
    FuncFit[3*n:3*n+3] = 1.

for n in range (nexp,nfunc):
    FuncFit[3*n:3*n+3] = 0.

FuncFit = [2,2,2,1,1,1,0,0,0]

Avector = np.zeros(len(FuncFit),dtype=np.longdouble)
for k in range(2,len(FuncFit),3):
    Avector[k]=1.

del n,k,nfunc,npoly,nexp

# triple length Xvect
Xvect = np.append(Xvect,Xsave)


Asolv7, chi_sq7,covar7,alpha7 = rfm.mrqmin(Avector,Xvect,Yvect,Yerr,FuncFit,error)

XtripleTrue = np.append(XdoubleTrue,Xtrue)

Yfunc7 = np.zeros(len(XtripleTrue),dtype=np.longdouble)
dydx7 = np.zeros([len(XtripleTrue),len(Avector)],dtype=np.longdouble)

Yfunc7, dydx7 = rfm.grand_func(Asolv7,XtripleTrue,FuncFit,error)

Ypoints7 = np.zeros(len(Xsave),dtype=np.longdouble)
dydum7 = np.zeros([len(Xvect),len(Avector)],dtype=np.longdouble)

Ypoints7,dydum7 = rfm.grand_func(Asolv7,Xvect,FuncFit,error)


fig7 = plt.figure()
plt.plot(Xsave,Yvect,'bo')
plt.plot(Xtrue,Yfunc7,'blue')

plt.plot(Xtrue,Ytrue,'black')


#plt.suptitle('Fitting Forms')
plt.suptitle('Solutions 3 var (black=actual):\nblue=poly+gauss+expt')
plt.axis([x1,x2,y1,y2])
plt.show()

plt.savefig('fit_3var.png')
# create scatter plot

fig4, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

ax1.scatter(Ypoints1, Yvect, s=20, c='blue')
ax1.axis([y1,y2,y1,y2])
ax1.plot(ax1.get_xlim(), ax1.get_ylim(), color='gray')
ax1.plot([y1,y1],[y2,y2],color='gray')
ax1.set_ylabel('Obs')
ax1.set_title('1 poly')

ax2.scatter(Ypoints2, Yvect, s=20, c='green')
ax2.axis([y1,y2,y1,y2])
ax2.plot(ax2.get_xlim(), ax2.get_ylim(), color='gray')
ax2.set_title('gauss')

ax3.scatter(Ypoints3, Yvect, s=20, c='red')
ax3.axis([y1,y2,y1,y2])
diag_line, = ax3.plot(ax3.get_xlim(), ax3.get_ylim(), color='gray')
ax3.set_title('expt')


ax4.scatter(Ypoints4, Yvect,s=20, c='blue')
ax4.axis([y1,y2,y1,y2])
ax4.plot(ax4.get_xlim(), ax4.get_ylim(), color='gray')
ax4.set_xlabel('Model')
ax4.set_ylabel('Obs')
ax4.set_title('poly + gauss')


ax5.scatter(Ypoints5, Yvect,s=20, c='red')
ax5.axis([y1,y2,y1,y2])
ax5.plot(ax5.get_xlim(), ax5.get_ylim(), color='gray')
ax5.set_xlabel('Model')
ax5.set_title('poly+expt')

ax6.scatter(Ypoints6, Yvect,s=20, c='green')
ax6.axis([y1,y2,y1,y2])
ax6.plot(ax6.get_xlim(), ax6.get_ylim(), color='gray')
ax6.set_xlabel('Model')
ax6.set_title('gauss + expt')

plt.show()
plt.savefig('scatter.png')


fig9 = plt.figure()
plt.axis([y1,y2,y1,y2])
plt.plot([y1,y2],[y1,y2],color='gray',linestyle='solid')
plt.scatter(Ypoints7,Yvect)


plt.xlabel('Model')
plt.ylabel('Obs')
plt.title('exp + gauss + poly')

plt.show()
plt.savefig('scatter7.png')

print 'Done'
