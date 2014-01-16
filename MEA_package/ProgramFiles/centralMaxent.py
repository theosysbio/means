#!/project/soft/linux64/epd/bin/python

import numpy
import math
import sys
import time
from numpy import linalg
from scipy import integrate, optimize, stats, misc
from decimal import *
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import argparse

##############################################################################
# USAGE:
# bash$ python centralMaxent.py [in_file] [out_file]
# where in_file is a list of the moments output
# out_file is where you want to write to
##############################################################################

##############################################################################
#return the moments up to power n
def sampleMoments(x, n):
    sampleMoment = []                # list of central moments for storage
    for i in range(1,n+1):
        a=0                           # initialise cumulative central moment
        for j in range(0,len(x)):
            a = a + ((x[j])**i)/len(x) # add a central moment term
        sampleMoment.append(a)
    return sampleMoment

def calcSampleMoment(x, i):
    sampleMoment = 0
    a = 0
    for j in range(0,len(x)):
        a = a + ((x[j])**i)/len(x) # add a moment contribution
    sampleMoment = a
    return sampleMoment

##############################################################################

#vecLambda = numpy.zeros(shape=(maxOrderMoment),dtype=numpy.float128) # initialise vector of Lagrange multipliers

# stretch all the moment constraints such that the variance equals unity
def stretchConstraints(dctCenMoments, lim_lower, mean):
    dctStretchedConstraints={}
    for key in dctCenMoments:
        dctStretchedConstraints[key]=dctCenMoments[key]/(dctCenMoments[2]**(key/2))

    lim_lower=(lim_lower-mean)/dctCenMoments[2]

    return (dctStretchedConstraints, lim_lower)
    
# initial lambda are set to reflect moments of standard Gaussian distribution
def initialiseLambda(dctCenMoments):
    
    vecLambda=[]
    i=1
    for moment in dctCenMoments:
        if i%2==0:
            vecLambda.append(-1/misc.factorial2(i-1))
        else:
            vecLambda.append(0)
        i+=1

    return vecLambda

def stretchLambda(dctCenMoments, vecLambda):
    
    vecNewLambda=[]
    i=0
    for moment in dctCenMoments:
        vecNewLambda.append(vecLambda[i]/(dctCenMoments[2])**(moment/2))
        i+=1

    return vecNewLambda

def setLambda(lstLambda, threshold, maxOrderMoment):
    
    if len(lstLambda)>=2:
        diff=[]
        for k in range(len(lstLambda[0])):
            diff.append(abs(lstLambda[len(lstLambda)-1][k]-lstLambda[len(lstLambda)-2][k]))
        if max(diff)<threshold*1000:
            #print "Maximum difference is " + str(max(diff)) + "\nRecycling previous Lagrange multipliers"
            vecLambda=lstLambda[len(lstLambda)-1]
        else:
            vecLambda=numpy.zeros(maxOrderMoment)
            #for ik in range(len(vecLambda)):
                #vecLambda[ik]=10**(10*-ik)
    else:
        vecLambda=numpy.zeros(maxOrderMoment)
        #for ik in range(len(vecLambda)):
            #vecLambda[ik]=10**(10*-ik)
    
    return vecLambda

        

###############################################################################
# evaluate the maximum entropy distribution for a given order of moments
# at a given value of x
# takes x, maximum moment and vector of Lagrange multipliers and input

def momTerm(x,n,vecLambda):
    xvec=[]
    for i in range(1,maxOrderMoment+1):
        xvec.append(numpy.float128(-(x)**i))
    xvec = numpy.array(xvec,dtype=numpy.float128)
    a = numpy.dot(xvec,vecLambda)
    try:
        a = math.exp(a)*x**n
    except OverflowError:
        plotIntegrand(n, vecLambda)
        print "Maxent failed because the numerical integration algorithm found a divergent integrand. Exiting. The integrand has been plotted and can be found in the working directory named maxent_out_failed_distn.pdf"
        print "Failure occurred at time point " + str(timestep+1)
        print "Vector of Lagrange multipliers at failure = ", vecLambda
        sys.exit()
    return a

def plotTerm(x, n, vecLambda):
    xvec=[]
    for i in range(1,maxOrderMoment+1):
        xvec.append(numpy.float128(-(x)**i))
    xvec = numpy.array(xvec,dtype=numpy.float128)
    a = numpy.dot(xvec,vecLambda)
    
    try:
        a = math.exp(a)*x**n
        if a > 1e6:
            a = 1e6
    except OverflowError:
        if a > 0:
            a = 1e4
        else:
            a = -1e4
    return a

def plotIntegrand(n, vecLambda):
    print vecLambda
    x=[]
    y1=[]
    y2=[]

    x = pylab.arange(lim_lower, lim_upper, 0.01)
    for x_i in x:
        y1.append(plotTerm(x_i, n, vecLambda))
        y2.append(plotTerm(x_i, 0, vecLambda))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(x, y1, c='r', linestyle="solid")
    ax1.plot(x, y2, c='k', linestyle="solid")
    
    plt.xlabel("$x$")
    plt.ylabel("$f(x)=\exp(-\sum_{i=1}^M \, \lambda_{i}x^{i})$")
    plt.ylim([0,10])
    plt.xlim([lim_lower/10, -lim_lower/10])
    plt.savefig("maxent_out_failed_distn.pdf")

###############################################################################
# potential function
###############################################################################

def potentialFn(vecLambda, dctStretchedConstraints, lim_lower, lstConstraints):
    gamma=math.log(momentCalc(0, vecLambda, lim_lower)[1])+numpy.dot(vecLambda, numpy.array(lstConstraints))
    return gamma

###############################################################################
# numerical integration to yield maximum entropy distribution moment estimate
# n is the order of moment required
# vecLambda is the array of Lagrange multipliers (one for each moment)
# lim_lower and lim_upper are the limits of integration
# m is the number of domains you wish to split the quadrature integration into

def momentCalc(n, vecLambda, lim_lower):
    momNumerator=numpy.float128(0)
    momNormTerm=numpy.float128(0)
    momNumerator+=integrate.quad(momTerm,lim_lower,lim_upper, args=(n, vecLambda))[0] # get the numerator
    momNormTerm+=integrate.quad(momTerm, lim_lower, lim_upper, args=(0,vecLambda))[0] # get the normalisation constant
    return momNumerator / momNormTerm, momNormTerm # return moment and normalisation term

###############################################################################
# Solution via Legendre Transformation
# 0) Verify the existence of a solution via the condition det(Hankel matrix)>0
# 1) Define the dual potential function as G = log Z + lambda[n].mu^n
# 2) Find the gradient of the potential function with respect to lambda(n)
#    as (mu(n) - <x**n>)
# 3) Find the Hessian matrix <x**(n+m)> - <x**n>*<x**m>
# 4) Invert the Hessian
# 5) Update: new_lambda[n] = old_lambda[n] - <invH[n,m],grad[m]>
# 6) Repeat steps 2-5 until maximum change in any Lagrange multiplier is
#    less than the user-defined threshold
###############################################################################


###############################################################################

# Calculate and invert the Hessian
def calcInvHessian(vecLambda, dctStretchedConstraints, lim_lower, lstConstraints):
    H=numpy.zeros(shape=(maxOrderMoment,maxOrderMoment))
    for i in dctStretchedConstraints:
        for j in dctStretchedConstraints:
            H[i-1][j-1]=momentCalc(i+j,vecLambda,lim_lower)[0] - momentCalc(i,vecLambda,lim_lower)[0] * momentCalc(j,vecLambda,lim_lower)[0] # calculate the terms of the Hessian as <x**(i+j)> - <x**i>*<x**j>
    # invH = linalg.inv(H) # invert the Hessian
    return H

# Calculate the gradient for Lagrange multiplier i=1,...,n
# returns a vector rank n
def calcGradient(vecLambda, dctStretchedConstraints, lim_lower, lstConstraints):
    vecGradient = []
    for i in dctStretchedConstraints:
        vecGradient.append(dctStretchedConstraints[i]-momentCalc(i,vecLambda,lim_lower)[0]) 
    return numpy.array(vecGradient)

##############################################################################
# USER INPUT PARAMETERS

start = time.clock()

'''
try:
    in_filename = sys.argv[1]
except:
    "Please specify a valid input file name."
try:
    out_filename = sys.argv[2]
except:
    "Please specify a valid output file name."

in_file=open(in_filename,"r")
out_file=open(out_filename,"w")
'''

##################################
# parser for run time arguments
##################################

parser = argparse.ArgumentParser(description='Compute maximum entropy distribution.')
parser.add_argument('inputFile', metavar='I')
parser.add_argument('outputFile', default='batchMaxent_out.dat', metavar='O')
parser.add_argument('--graphs', required=False, default='n', help='y to plot graphs', choices=['y','n'])
parser.add_argument('--min_method', required=False, default='b', help='b to use BFGS; n to use Newton-CG', choices=['b','n'])

dctRuntimeArgs = vars(parser.parse_args())

try:
    in_file=open(dctRuntimeArgs['inputFile'], "r")
except:
    print "Please specify a valid input file name."

out_file=open(dctRuntimeArgs['outputFile'], "w")


lstTimePoints=[]
lstConstraints=[]
lstLambda=[]
lstZ=[]
while True:
    in_line = in_file.readline()
    if in_line=="":
        break

    if in_line[0]!=">":
        if in_line[0:4]=="time":
            in_line=in_line.rstrip()
            lstTimePoints=in_line.split("\t") # split in_line using tab as delimiter
            del lstTimePoints[0] # remove "time" word from list
            del lstTimePoints[0] # remove initial conditions
        else:
            in_line=in_line.rstrip()
            lstLine=in_line.split("\t")
            del lstLine[0] # remove m# from list
            del lstLine[0] # remove initial conditions (no need to estimate these!)
            lstConstraints.append(lstLine)
        

arrConstraints=numpy.array(lstConstraints)
arrConstraints=numpy.transpose(arrConstraints)

lstMoments=[]
meanCount=0
lstMeans=[]
timestep=0
for constraintSet in arrConstraints:
    lstConstraints=[]
    
    mean=0
    dctCenMoments={}
    for j in range(len(constraintSet)):
        if j==0:
            mean=float(constraintSet[0])
            lstMeans.append(mean)
            dctCenMoments[j+1]=0 # central moment so mean is zero
            lstConstraints.append(0)
        else:
            dctCenMoments[j+1]=float(constraintSet[j])
            lstConstraints.append(float(constraintSet[j]))

    #print mean
    #print lstConstraints

    maxOrderMoment = max(dctCenMoments) # maximum order of moments
    threshold = 1e-9 # threshold for convergence in optimisation
    lim_lower = -mean # initial limits of integration
    lim_upper = max([(lstConstraints[1]**0.5)*20-mean,10])
    quadDivisions = 1 # number of regions to divide function into for integration

    dctStretchedConstraints=dctCenMoments
    vecLambda=setLambda(lstLambda, threshold, maxOrderMoment)
    #vecLambda=numpy.zeros(shape=maxOrderMoment)
###############################################################################
# Verify existence conditions
# Determinant of Hankel matrix must be positive for a solution to be found
# Currently uses moment truncation - any moment outside of range of specified
# moments is preset to zero

###############################################################################
    '''
    hankelMatrix=numpy.zeros(shape=(maxOrderMoment, maxOrderMoment))
    if maxOrderMoment % 2==0:
        for i in range(1,maxOrderMoment+1):                         # even moments
            for j in range(1,maxOrderMoment+1):
                if i+j>maxOrderMoment:
                    hankelMatrix[i-1,j-1]=0
                else:
                    hankelMatrix[i-1,j-1]=dctStretchedConstraints[i+j]
    else:                                                         # odd moments
        for i in range(1,maxOrderMoment+1):
            for j in range(1,maxOrderMoment+1):
                if i+j+1>maxOrderMoment:
                    hankelMatrix[i-1,j-1]=0
                else:
                    hankelMatrix[i-1,j-1]=dctStretchedConstraints[i+j+1]

    detHankelMatrix = linalg.det(hankelMatrix)
    if detHankelMatrix<0:
        print "Solution existence condition not satisfied. Execution terminated."
        sys.exit()
    else:
        print "Solution existence condition satisfied. Determinant of Hankel matrix is ", detHankelMatrix
        '''
    #vecLambdaPrime = numpy.zeros(shape=(maxOrderMoment),dtype=numpy.float128) # initialise updated Lagrange multiplier vector

# (5-6) Iterate on the Lagrange multiplier to find minimum
    '''
    diff=1
    while diff > threshold:
        invH = calcInvHessian(vecLambda)
        vecGradient = calcGradient(vecLambda, dctStretchedConstraints)
        vecLambdaPrime = vecLambda - numpy.dot(invH, vecGradient) # matrix product of invHessian with gradient vector

        diff = max(abs(vecLambdaPrime-vecLambda))
    
        print "vecLambdaPrime = ", vecLambdaPrime
        print "maxDiff = ", diff
        vecLambda = vecLambdaPrime
        '''
###############################################################################
# Use the Newton-CG method to minimise Gamma
###############################################################################

    #print "BFGS minimisation routine starting..."
    vecLambdaInitial=vecLambda
    #print dctStretchedConstraints
    #print lstConstraints

    if dctRuntimeArgs['min_method']=='n':
        lambdaOpt=optimize.fmin_ncg(potentialFn, vecLambdaInitial, calcGradient, fhess=calcInvHessian,args=(dctStretchedConstraints, lim_lower, lstConstraints),avextol=threshold, disp=0)
    elif dctRuntimeArgs['min_method']=='b':
        lambdaOpt=optimize.fmin_bfgs(potentialFn, vecLambdaInitial, calcGradient, args=(dctStretchedConstraints, lim_lower, lstConstraints),gtol=threshold, disp=0)

    #print lambdaOpt

###############################################################################
# Check output for consistency of moments
###############################################################################

    momentCheck=[]
    #print lambdaOpt

    for i in range(len(vecLambda)):
        momentCheck.append(momentCalc(i+1,lambdaOpt,lim_lower)[0])
    # calculate maximum entropy moment list

    #print "Check moments are equal:"
    #print "Max entropy moments = ", momentCheck
    #print "True moments = ", dctStretchedConstraints

##############################################################################
# Plot maximum entropy density function and histogram and/or
# true density over support
##############################################################################

    lstLambda.append(lambdaOpt)
    lstMoments.append(momentCheck)
    
    xD1=[]
    pME=[]
    normConstant=momentCalc(0,lambdaOpt,lim_lower)[1] # find the normalisation constant
    lstZ.append(math.log(normConstant))

    if dctRuntimeArgs['graphs']=='y':
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in range(0,1000):
            xD1.append(float(i)*(lim_upper-lim_lower)/1000)
            pME.append(momTerm(xD1[i]-mean,0,lambdaOpt)/1000)

        ax1.plot(xD1, pME, c='r', linestyle="solid")
        plt.title("Maximum entropy distribution at t=" + str(lstTimePoints[timestep]))
        plt.xlim(mean-5*(lstConstraints[1])**0.5,mean+5*(lstConstraints[1])**0.5)
        plt.savefig(str(lstTimePoints[timestep]) + ".png")

    timestep+=1

in_file.close()

##############################################################################
# Write output lambda vectors to text

#print lstZ
#print lstTimePoints
#print lstMeans

l_string=""
for i in range(1,len(lstLambda[0])+1):
    l_string+="l"+str(i)+"\t"

out_file.write("t\tMean\t"+"logZ\t"+ l_string+"\n")

for i in range(len(lstTimePoints)):
    out_file.write(str(lstTimePoints[i]) + '\t' + str(lstMeans[i]) + '\t' + str(lstZ[i]) + '\t')
    for j in lstLambda[i]:
        out_file.write(str(j)+'\t')
    out_file.write('\n')

out_file.close()

end=time.clock()

print "Maxent terminated successfully with a processing time of " + str(round(end-start,1)) + " seconds for " + str(len(lstZ)) + " time points."

##############################################################################
# 
##############################################################################

'''
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
delta = 0.00001
x = np.arange(-0.1, 1.0, delta)
y = np.arange(0, 1.0, delta)

for x_i in x:
    for y_i in y:
        vl=[x_i,y_i]
        z=potentialFn(vl, 

# difference of Gaussians
Z = 10.0 * (Z2 - Z1)
'''
