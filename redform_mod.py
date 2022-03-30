#!/bin/python

import numpy as np

import matplotlib.pyplot as plt

tiny = np.float64(1.e-6)  # small number to prevent divide by zero
tiny_sq = tiny*tiny

def grand_func(Avect,Xvect,FuncFit,error):

#  pass-thru routine which calls function of form FuncFit

#    Amat is a the row of the A coeffcients of size 3*n
#    Xvect is the variables, [X1,X2,...,Xn] where X1=[x11,x12,...x1n]
#                                                 X2=[x21,x22,...x2n]

#    deriv are the 1st derivatives
#    FuncFit is a vector containing fitting forms in sets of 3
#       0 = poly
#       1 = exponential
#    Return error = True if a function is not defined
#    Return Xsum as value of the fitting function (vector)
#    Return deriv as value of 1st derivaties (matrix)

# determine # of functions
    nfunc_gf = int(len(FuncFit)/3)
# deduce length of each X
    index_gf = int(len(Xvect)/nfunc_gf)

#   print('nfunc_gf',nfunc_gf,type(nfunc_gf))
#   print('index_gf',index_gf,type(index_gf))
 

    Xsum = np.zeros(index_gf,dtype=np.float64)
    dfdx = np.zeros((index_gf,len(Avect)),dtype=np.float64)

#    print 'FuncFit',FuncFit

# this needs to be increment Avect in chunks of 3 (Avector[i:i+3])
# but Xx by 1 ...

    for i_gf in range(0,index_gf):

## ii is a pointer to the ith element of each X1[i],X2[i],contained in the long
##    vector Xvect

        Xval=0.

        for j_gf in range(0,len(FuncFit),3):
            ii_gf = i_gf+j_gf*int(index_gf/3)
            pass

#           print('ii_gf',ii_gf,type(ii_gf))

            if (FuncFit[j_gf]==0):
                Xval, dfdx[i_gf][j_gf:j_gf+3] = poly_func(Avect[j_gf:j_gf+3],Xvect[ii_gf])
            elif (FuncFit[j_gf]==1):
                Xval, dfdx[i_gf][j_gf:j_gf+3] = exp_func(Avect[j_gf:j_gf+3],Xvect[ii_gf])
            elif (FuncFit[j_gf]==2):
                Xval, dfdx[i_gf][j_gf:j_gf+3] = gauss_func(Avect[j_gf:j_gf+3],Xvect[ii_gf])

            else:
                print('Function ',FuncFit,' not defined')
                error = True

            Xsum[i_gf] += Xval

#        print('i_gf: ',i_gf,Xsum[i_gf])

    del i_gf,ii_gf,j_gf,Xval,nfunc_gf,index_gf


    return Xsum,dfdx

def poly_func(Avector,Xx):
#    polynomial function
#    Expects Avector as a 3-element vector
#    Expects Xvector as a scalar

#    ReturnsPolySum = value
#    Returns dpda = 1st derivative (vector)

    dpda = np.zeros(3,dtype=np.float64)
#
    PolySum = Avector[0]*Xx + Avector[1]*Xx*Xx + Avector[2]*Xx*Xx*Xx
    dpda[0] = Xx
    dpda[1]= Xx*Xx
    dpda[2]= Xx*Xx*Xx

    return PolySum,dpda

def poly_init(expected_in,expected_out):

# for a given input and output, calculate coefficents for a linear polynomial
 #   GaussVect = np.zeros(3,dtype=np.float64)

    A = np.mean(expected_out,dtype=np.float64)
    B = 0.
    C = 0.

#    C = np.std(expected_in,dtype=np.double)*np.sqrt(np.pi)

    PolyVect = np.asarray([A,B,C],dtype=np.float64)

    return PolyVect


def exp_func(Avector,Xx):
#    exponential function
#    Avector is a vector of length 3
#    Xx is a scalar
#    dxda = 1st derivative vector

# make a local copy of the Avector
    Aloc = np.copy(Avector)
    dxda = np.zeros(3,dtype=np.float64)

# and set Avector[2] = tiny_sq if it is zero
    if (abs(Aloc[2]) < (tiny_sq) ):
        Aloc[2] = tiny_sq
# Use equation 1 from Lynch et al 2008

# for clarity
    A = Aloc[0]
    B = Aloc[1]
    C = Aloc[2]

## y = A*exp((X-B)/C)

    ExpTerm = np.exp(-((Xx-B)/C))


    ExpSum = A*ExpTerm
# and the derivative (courtesy of Mathematica)
    dxda[0] = ExpTerm
    dxda[1] = ExpTerm*(A/C)
    dxda[2] = ExpTerm*(A*(Xx-B))/(C*C)

    return ExpSum,dxda



def gauss_func(Avector,Xx):
#    gauss function
#    Expects Avector as a 3-element vector
#    Expects Xvector as a scalar

#    ReturnsPolySum = value
#    Returns dpda = 1st derivative (vector)

    Aloc = np.copy(Avector)

    # for clarity
    A = Aloc[0]
    B = Aloc[1]
    C = Aloc[2]

    dgda = np.zeros(3,dtype=np.float64)

# and set Avector[2] = tiny_sq if it is zero
    if (abs(C) < tiny_sq ):
        C = tiny_sq

# Eqn (15.5.16) from Numerical Recipes

    arg = (Xx-B)/(2.*C)

    GaussSum = A*np.exp(-arg**2.)

    dgda[0]= np.exp(-arg**2.)
    dgda[1]= np.exp(-arg**2.)*(A*(Xx-B))/(2.*C**2.)
    dgda[2]= np.exp(-arg**2.)*(A*(Xx-B)**2.)/(2.*C**3.)

    return GaussSum,dgda

def gauss_init(expected_out,expected_in):

# for a given input and output, calculate coefficents for a Gaussian distribution
    GaussVect = np.zeros(3,dtype=np.float64)

    A = np.mean(expected_out)

    sigma = 1./(A * np.sqrt(np.pi))

    B = np.mean(expected_in)

    C = sigma

    GaussVect = [A,B,C]

    return GaussVect
####

#def mrqcof(Xvect,Yvect,Yerr,Amat_loc,FuncFit,error):

def mrqcof(Avect,Xvect,Yvect,Yerr,FuncFit,error):

# calls grand_func to evaluate A*x=Ymod

# returns alpha(matrix),beta(vector),chi_sq(scalar)

#Xvect & Yvect are data being fitted
#Yerr is the difference between FuncSum and Yvect, ie Sigma)
#FuncFit determines the form of the fitting functin as a combination
# of available forms in grand_func()
# Amat_loc is a local copy of Amatrix
# alpha is the covariance matrix
# beta is the 2nd derivative of diagonal elements eqn 15.5.8 in Numerical Recipes
# chisquared is the return value, and error is passed upward if there is an error

#Ymod is the model value based on the fitting function
    Ymod = np.zeros(len(Xvect),dtype=np.float64)

#initialize alpha[m,m] and beta[m] as zero

    chi_sq = 0.
    weight = np.zeros(len(Xvect),dtype=np.float64)

    alpha = np.zeros((len(Avect),len(Avect)),dtype=np.float64)
    beta = np.zeros(len(Avect),dtype=np.float64)
    dyda = np.zeros((len(Xvect),len(Avect)),dtype=np.float64) # 1st derivative

    delta_y = np.zeros(len(Yvect),dtype=np.float64)


### then Y(x) = A1(x) + A2(x) + ...
    Ymod, dyda = grand_func(Avect,Xvect,FuncFit,error)
    pass

# calculate error between model and actual values
    delta_y[:] = Yvect[:]-Ymod[:]

# sigma-squared is 1/err^2
    sig_squared = 1./(Yerr*Yerr)

## this loop is trying to do eqn (15.5.11)
    for l in range(len(Avect)):
        weight = dyda[:,l]*sig_squared  # multiply dyda by sigma for all X
        for m in range(l+1):
## need to get these coefficients figured out...
            alpha[l,m] = np.dot(weight,dyda[:,m])
            alpha[m,l] = alpha[l,m]

        beta[l] = np.dot(delta_y,weight)

    chi_sq = np.dot(delta_y**2,sig_squared)
#
### Try eqn 15.5.11 from f77 (NR  Sect15.5
#    for i in range(len(Xvect)):
#        # function call would go here
#        # sig2 would be here
#        # delta_y calculated here
#        j = 0
#
#        for l in range(len(Avect)):  # iterating over Avect
#            weight[i] = dyda[i][l]*sig_squared[i]
#            k = 0
#            for m in range(0,l+1):
#
#                alpha[l][m] = alpha[l][m] + weight[i]*dyda[i][m]
#                k =+ 1
#            beta[l]=beta[l]+delta_y[i]*weight[i]
#            j =+ 1
#        chi_sq = chi_sq + delta_y[i]*delta_y[i]*sig_squared[i]
#
#    for j in range(1,len(Avect)):
#        for k in range(j):
#            print(j,k)
#            alpha[k,j]=alpha[j,k]


    if (error == True):
        print('Fitting Function Failed')
    else:
#        print('Fitting Function Value:',Ymod)
        pass

    return alpha,beta,chi_sq

# in NR this is part of mrqmin but
#def mqq_init(






def gaussjordan(Amat,Bvect):
# Guass-Jordan with full pivoting
# Input Amat (matrix) and Bvect (vector)
# Returns Ainv (inverse of Amatrix)
#      and Bsolv (vector solution to Ax= B see NR eqn 2.1.1)

    ipivot = np.zeros_like(Bvect,dtype=int)  # the pivot index
    indx_r = np.zeros_like(Bvect,dtype=int)  # the row of the pivot index
    indx_c = np.zeros_like(Bvect,dtype=int)  # the col of the pivot index

# copy Amat to local array
    Aloc = np.copy(Amat)
    Bloc = np.copy(Bvect)

    Ainv = np.copy(Amat)
    Bsolv = np.copy(Bvect)

# big do loop, see NR section 2.1
    for i in range(len(Bvect)):    # looping over columns

        biggest = 0.

        for j in range(len(Bvect)):  # loop over rows
            if (ipivot[j] != 1):
                for k in range(len(Bvect)):   # loop over columns
                    if (ipivot[k] == 0):
                        if (abs(Aloc[j,k]) >= biggest):
                            biggest = abs(Aloc[j,k])
                            irow = j
                            icol = k


# this is the pivot element, on column icol
        ipivot[icol] = ipivot[icol]+1

# there is some language in the f77 routine that says "trust me"...
# ... having no other choice, pivot the row,
# ... if irow ne icol there will be an interchange of columns:

        if (irow != icol):
            for ii in range(len(Bloc)):
                dum = Aloc[irow,ii]
                Aloc[irow,ii] = Aloc[icol,ii]
                Aloc[icol,ii] = dum

# the second do loop would be if B was a matrix rather than a vector
# commented out for now
#            for ii in range( [ M = #col of B)

# but still need to change order of B (if irow ne icol)
            dum = Bloc[irow]
            Bloc[irow] = Bloc[icol]
            Bloc[icol] = dum

# Now divide the pivot row by the pivot element, which is Aloc[icol,irow]

# Store the values for i (the outermost loop)
        indx_r[i] = irow
        indx_c[i] = icol

# now divide by the pivot element
        if (Aloc[icol,icol] == 0.):
            error = True
            print('Warning: singular matrix in Gauss-Jordan')
#           Ainv[:][:] = -999
#            Bsolv[:] = -999
#
# set pivot element to tiny value
#            Aloc[icol][icol]=tiny_sq
            Aloc[icol,icol] = 1

#           return Ainv,Bsolv

        pivinv = 1./Aloc[icol,icol]
        Aloc[icol,icol]=1.

        for ii in range(len(Bvect)):
            Aloc[icol,ii] = Aloc[icol,ii]*pivinv

# again would have a do loop for B
# but M=1 so divide B[icol]/A[icol,icol]
        Bloc[icol] = Bloc[icol]*pivinv

# now reduce the nonpivot rows
        for ii in range(len(Bloc)):
            if (ii != icol):  # not the pivot one
                dum = Aloc[ii,icol]
                Aloc[ii,icol]=0.
                for jj in range(len(Bloc)):
                    Aloc[ii,jj] = Aloc[ii,jj] - Aloc[icol,jj]*dum
# M = 1 but a loop would go here for B
                Bloc[ii] = Bloc[ii] - Bloc[icol]*dum


# if we get here we're done, need to unscramble the interchanges
# (this is the part that said "trust me")
    for i in range(len(Bloc)-1,-1,-1):  # working backwards through the list
        if (indx_r[i] != indx_c[i]):
            for k in range(len(Bloc)):
                dum = Aloc[k,indx_r[i]]
                Aloc[k,indx_r[i]] = Aloc[k,indx_c[i]]
                Aloc[k,indx_c[i]] = dum

# Set Ainv = Aloc & Bsolv = Bvect here
# The rest of the routine modifies those values
    Ainv = Aloc
    Bsolv = Bloc


    return Ainv, Bsolv


def mrqmin(Avect,Xvect,Yvect,Yerr,FuncFit,error):

# Levenberg-Marquart Solver
# Input: Xvect, Yvect, Sigma are input, output, stddev/erro (vectors, constant)
#        Amat_loc is the local copy of the A matrix
#        FuncFit is a Vector of what fitting forms (in sets of 3)
#
# Output: Anew, Covar, alpha are matrices
#         beta, delta_A... vectors
#         chi_sq, lamda_new  scalars

# all passes:
#        set covar = alpha
#       multiply covar diagonals by lamda
#       set delta_A = beta
#       call gauss-jordan on (covar,delta_A)
#       call mrqcof  with current value of Atry
#       compare chisquared
#       if chisquared < old value:
#            lamda = lamda/10
#             old chisq = chisquared
#             alpha = covar
#             beta = delta_A
#             keep Atry (Amatrix_new = Atry)
#       otherwise
#            lamda = lamda*10
#             chisq = old chisquared
#
# last pass
#       set covar = alpha
#       delta_A = beta
#       call gauss-jordan on (covar, delta_A)
#       alpha not updated, is curvature matrix
#       covar is covariance

# stopping condition:
#       chsqquare decreases by small amount (0.01 or 0.001)
#       not chisquare increasing by any amount


#alpha doubles as the covariance matrix
    alpha = np.zeros((len(Avect),len(Avect)),dtype=np.float64)
    covar = np.zeros((len(Avect),len(Avect)),dtype=np.float64)

# beta, or da, is local to mrqmin)
    beta = np.zeros(len(Avect),dtype=np.float64)
    delta_A = np.zeros(len(Avect),dtype=np.float64)


### mrq min in NR does a lot of things... simplify?
# first pass = set lamda = 0..1
#           call mrqcof to get alpha, beta, chisquared
#           sets local Atry = Amatrix where Amatrix is the initial array
#              (not sure how that works out when fitting real data...)

# initialize mrq process
    lamda = 0.001

#set chi_sq
#    chi_sq = 1.

# Amat_loc is the local copy of Avect (preserve Avect)
    Amat_loc = np.copy(Avect)


# get initial alpha and beta from mrqcof
    print('initializing...')
    alpha,beta,chi_sq = mrqcof(Amat_loc,Xvect,Yvect,Yerr,FuncFit,error)
#    print('MRQ IN: ', Amat_loc,Yerr)
#    print('MRQ OUt ',alpha,beta,chi_sq)

#Copy Amatrix to Alast
    Atry = np.copy(Amat_loc)


#And keep track of chi_squared
    chi_min = np.copy(chi_sq)
    chi_last= np.copy(chi_sq)

    ntry = 0
    nmax = 1000

    for n in range(0,nmax):

        ntry = ntry + 1

## save previous value of chi_sq
        chi_last = chi_sq

# set covar = alpha ... from f77

        covar[:][:] = alpha[:][:]
        delta_A[:] = beta[:]

        for j in range(len(Avect)):
            covar[j][j] = covar[j][j]*(1.+lamda)

## and multiply the diagonal by 1 + lamda
#       covar = np.copy(alpha)
#        for jj in range(len(Avect)):
#            dum = covar[jj,jj]
#            covar[jj,jj] = dum*(1.+lamda)
#
#        delta_A = np.copy(beta)

# call gauss-jordan with the updated covariance and delta_A
        covar_dum, beta_dum = gaussjordan(covar,delta_A)

# and assign the returned values to covar and delta_A
        covar = np.copy(covar_dum)
        delta_A = np.copy(beta_dum)

# check for lamda = 0 (converged)
        if (lamda == 0.):
            print('lamda = 0.')
            print('chi_sq=',chi_sq, chi_last, chi_min)
            return Amat_loc,chi_sq,covar,alpha

        Atry = Amat_loc + delta_A

# then call MRQCOF to get new alpha, chi_sq
        covar, delta_A, chi_sq = mrqcof(Atry,Xvect,Yvect,Yerr,FuncFit,error)
#        print 'MRQ IN: ', Atry,Yerr
#        print 'MRQ OUt ',covar,delta_A,chi_sq

        print('Try: ',ntry,chi_sq, chi_last,chi_min, lamda)

# see if chi-sq decreased
        if (chi_sq < chi_min):

## yes, then accept Atry
            lamda = 0.1 * lamda
            alpha = np.copy(covar)
            beta = np.copy(delta_A)
            Amat_loc = np.copy(Atry)

# now check for convergence
            if ((chi_sq <= chi_min) and (abs(chi_sq - chi_min) <= np.sqrt(tiny))):
                print('Converged by small decrease')
                print('Converged: ntries =',ntry)
                print('chi sq, chi last:',chi_sq,chi_min)
                print('chi_last:',chi_last)
                lamda = 0.0

            chi_min = chi_sq

        else:
            lamda = 10.*lamda

            if ( (chi_sq <= chi_last) and (abs(chi_sq - chi_last) <= tiny)):
                print('Converged by machine error:',tiny)
                print('Converged: ntries =',ntry)
                print('chi_sq=',chi_sq, chi_last)
                print('chi_min =',chi_min)
                Amat_loc = Atry
                lamda = 0.

            chi_sq = chi_min

        pass
#        print('Not converged yet')




    print('No convergence!')
    error = True
    print('chi_sq=',chi_sq,chi_last,chi_min, lamda)

    return Atry,chi_sq,covar,alpha
