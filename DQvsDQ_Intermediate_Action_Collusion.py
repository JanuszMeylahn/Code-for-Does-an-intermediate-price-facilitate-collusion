import numpy as np
import numba
from numba import njit
import math
import random
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import fsolve
from duopoly_functions import *
import time
from array import array

#Simulation paramters
NSamples = 500
NBatches = 31
InitialBatchK = 2000000
lincons = 0
FinalT = FinalTime(InitialBatchK, lincons, NBatches)
BatchsizeVectorinit = []
for i in range(0, NBatches-1):
     BatchsizeVectorinit.append(IncreasingBatchK(InitialBatchK, lincons, i))
BatchsizeVector  = np.array(BatchsizeVectorinit, dtype=int)
M = 3
a1 = 1
a2 = 1
c1 = 0
c2 = 0
mu = 4.
xi = 0.0
option = 'uniform' #Choosing the type of Q-matrix initialization. Options are std_normal, uniform, constant, constant_scaled or Calvano.
optionP = 'Calvano' #Choosing the type of price interval. Options are Calvano and SymmetricNash
alpha = 1.0
deltaQ = 0.95
inertia = 0.1
constanteps = 0.1


#Initialize environment
nashprices = FOCsim(1, 1, a1, a2, mu, c1, c2)
monprices = FOCsimJoint(2, 2, a1, a2, mu, c1, c2)
actionspaceinitial = np.array(actionspacegen(optionP, M, xi, nashprices, monprices), dtype=float)
statespaceinitial = np.array(statespacegen(M, actionspaceinitial), dtype=float)
priceinterval = np.array(priceintervalgen(optionP, xi, nashprices, monprices), dtype=float)
payoffs1 = np.array(payoffmatrix(0, M, actionspaceinitial, a1, a2, mu, c1, c2), dtype=float)
payoffs2 = np.array(payoffmatrix(1, M, actionspaceinitial, a1, a2, mu, c1, c2), dtype=float)
qmatrix1initial = np.array(Qinitialization(option, M, 0, deltaQ, payoffs1), dtype=float)
qmatrix2initial = np.array(Qinitialization(option, M, 1, deltaQ, payoffs2), dtype=float)

#Define random vectors to ensure that each sample trajectory uses its own sequence of random numbers
np.random.seed(974629)
randomvector1 = np.random.randint(1, 1000000, NSamples)
np.random.seed(109851)
randomvector2= np.random.randint(1, 1000000, NSamples)

#Tracked strategies
AllDstratinitial = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

#Open file for collecting data
file1 = open('AllD.txt', 'w')
file2 = open('SymmetricNE.txt', 'w')
file3 = open('CollusiveNE.txt', 'w')
file4 = open('ColLevel.txt', 'w')

#Start timer for recording how long the simulation took to run
start_time = time.time()

#The function performing the simulation
def QvsQsimulation(FinalT, qmatrix1initial, qmatrix2initial, actionspaceinitial, statespaceinitial, payoffs1, payoffs2, BatchsizeVector, deltaQ):
    revenuesamp1 = np.zeros(NSamples)
    revenuesamp2 = np.zeros(NSamples)
    Collist = np.zeros(int(NBatches)+1)
    ADlist= np.zeros(int(NBatches)+1)
    SymmetricNElist=np.zeros(int(NBatches)+1)
    CollusiveNElist=np.zeros(int(NBatches)+1)
    actionspacetemp = actionspaceinitial
    statespacetemp = statespaceinitial
    zerosM =list(np.zeros(M))
    actionspace = [zerosM, zerosM]
    statespace = []
    AllDstrat = []

    #Absorbing state strategies
    for listitA in range(M):
        actionspace[0][listitA] = round(actionspacetemp[0][listitA], 5)
        actionspace[1][listitA] = round(actionspacetemp[1][listitA], 5)
    for listitB in range(M*M):
        statespace.append([round(statespacetemp[listitB][0], 5), round(statespacetemp[listitB][1], 5)])
    for listitG in range(2*M*M):
        AllDstrat.append(AllDstratinitial[listitG])


    #Loop over the samples of the simulation
    for n in range(NSamples):
        #Initial Q-matrix
        if option == 'constant' or option == 'Calvano' or option == 'constant_scaled':
            qmatrix1act=np.zeros((M*M, M))
            qmatrix2act=np.zeros((M*M, M))
            qmatrix1value=np.zeros((M*M, M))
            qmatrix2value=np.zeros((M*M, M))
            for listitD in range(M*M):
                for listitE in range(M):
                    qmatrix1act[listitD][listitE] = qmatrix1initial[listitD][listitE]
                    qmatrix2act[listitD][listitE] = qmatrix2initial[listitD][listitE]
                    qmatrix1value[listitD][listitE] = qmatrix1act[listitD][listitE]
                    qmatrix2value[listitD][listitE] = qmatrix2act[listitD][listitE]
        
        elif option == 'std_normal':
            np.random.seed(randomvector1[n])
            qmatrix1act[listitD][listitE] = np.random.normal(0, 1, (M*M, M))
            np.random.seed(randomvector2[n])
            qmatrix2act[listitD][listitE] = np.random.normal(0, 1, (M*M, M))
            qmatrix1value=np.zeros((M*M, M))
            qmatrix2value=np.zeros((M*M, M))
            for listitD in range(M*M):
                for listitE in range(M):
                    qmatrix1value[listitD][listitE] = qmatrix1act[listitD][listitE]
                    qmatrix2value[listitD][listitE] = qmatrix2act[listitD][listitE]

        elif option == 'uniform':
            qmatrix1value=np.zeros((M*M, M))
            qmatrix2value=np.zeros((M*M, M))
            np.random.seed(randomvector1[n])
            qmatrix1act = np.random.uniform(0, 1, (M*M, M))
            np.random.seed(randomvector2[n])
            qmatrix2act = np.random.uniform(0, 1, (M*M, M))
            for listitD in range(M*M):
                for listitE in range(M):
                    qmatrix1value[listitD][listitE] = qmatrix1act[listitD][listitE]
                    qmatrix2value[listitD][listitE] = qmatrix2act[listitD][listitE]

        #Calculate current strategy
        strategypairvec=np.zeros(2*M*M)
        convergedstrat = np.zeros(2*M*M)
        for l in range(M*M):
            maxindex1 = np.argmax(qmatrix1act[l])
            maxindex2 = np.argmax(qmatrix2act[l])
            strategypairvec[l] =  maxindex1
            convergedstrat[l] = maxindex1
            strategypairvec[l+(M*M)] =  maxindex2
            convergedstrat[l+(M*M)] = maxindex2

        #Record Initial strategies
        matchSymmetricNE = 1
        matchAllD = 1
        matchCollusiveNE = 0
        #Check if the strategy is AllD
        for listitF in range(M*M):
            if int(convergedstrat[listitF]) != int(AllDstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(AllDstrat[listitF+(M*M)]):
                matchAllD = 0
        #Check if the srategy is symmetric 
        if int(convergedstrat[0]) != int(convergedstrat[9]) or int(convergedstrat[1]) != int(convergedstrat[12]) or int(convergedstrat[2]) != int(convergedstrat[15]) or int(convergedstrat[3]) != int(convergedstrat[10]) or int(convergedstrat[4]) != int(convergedstrat[13]) or int(convergedstrat[5]) != int(convergedstrat[16]) or int(convergedstrat[6]) != int(convergedstrat[11]) or int(convergedstrat[7]) != int(convergedstrat[14]) or int(convergedstrat[8]) != int(convergedstrat[17]):
            matchSymmetricNE = 0
        #Check if the symmetric strategy is collusive
        if matchSymmetricNE == 1 and int(convergedstrat[0]) == 1 and int(convergedstrat[4]) == 1:
            matchCollusiveNE = 1
        if matchSymmetricNE == 1 and int(convergedstrat[0]) == 2 and int(convergedstrat[8]) == 2:
            matchCollusiveNE = 1
        if matchSymmetricNE == 1 and int(convergedstrat[0]) == 1 and int(convergedstrat[4]) == 2 and int(convergedstrat[8]) == 2:
            matchCollusiveNE = 1


        #Record if it is one of the absorbing states
        if matchSymmetricNE == 1:
            SymmetricNElist[0] = SymmetricNElist[0] + 1./NSamples   
        if matchCollusiveNE == 1:
            CollusiveNElist[0] =  CollusiveNElist[0] + 1./NSamples
        if matchAllD == 1:
            ADlist[0] =  ADlist[0] + 1./NSamples 

        #Pricing history
        pricetraj1 = np.zeros(FinalT)
        pricetraj2 = np.zeros(FinalT)

        #Initialize time
        t = int(0)
        nbatch = int(0)

        #Random initial price/state
        randomactionindex1 = np.random.randint(0, len(actionspace[0]))
        randomactionindex2 = np.random.randint(0, len(actionspace[1]))
        pricetraj1[0] = float(actionspace[0][randomactionindex1])
        pricetraj2[0] = float(actionspace[1][randomactionindex2])
        s = [pricetraj1[0], pricetraj2[0]]

        
        #Initialize a vector for recording the strategy pair to all zeros
        strategypairvec=np.zeros(2*M*M) 

        #Loop over batches  
        while nbatch < NBatches: 

            #At the beginning of the batch transfer act to value matrix
            if nbatch > 1:
                for listitM in range(M*M):
                    for listitN in range(M):
                        qmatrix1value[listitM][listitN] = qmatrix1act[listitM][listitN]
                        qmatrix2value[listitM][listitN] = qmatrix2act[listitM][listitN]

            #Initialize batch data structures
            q1count = np.zeros((M*M, M))
            q2count = np.zeros((M*M, M))
            nbatch = nbatch + 1
            kbatch = int(0)
            collusioncount = 0.

            while kbatch < int(BatchsizeVector[nbatch-1]):
                t = t+1
                kbatch = kbatch + 1

                #Action-selection mechanism    
                random1 = np.random.uniform(0.0, 1.0)
                random2 = np.random.uniform(0.0, 1.0)
                sindex = statespace.index(s)
                if random1<constanteps:
                    randomactionindex1 = np.random.randint(0, len(actionspace[0]))
                    pricetraj1[t] = actionspace[0][randomactionindex1]
                else:
                    maxpos1 = np.argmax(qmatrix1act[sindex])
                    pricetraj1[t] = actionspace[0][maxpos1]

                if random2<constanteps:
                    randomactionindex2 = np.random.randint(0, len(actionspace[1]))
                    pricetraj2[t] = actionspace[1][randomactionindex2]
                else:
                    maxpos2 = np.argmax(qmatrix2act[sindex])
                    pricetraj2[t] = actionspace[1][maxpos2]

                #Update state
                s[0]  =  pricetraj1[t]
                s[1]  =  pricetraj2[t]
                s1index = actionspace[0].index(s[0])
                s2index = actionspace[1].index(s[1])

                #Batch data collection
                q1count[sindex][s1index] = q1count[sindex][s1index] + 1.
                q2count[sindex][s2index] = q2count[sindex][s2index] + 1.
                if q1count[sindex][s1index] > 1:
                    alphaaux = 1./(q1count[sindex][s1index]+1.)
                    qmatrix1value[sindex][s1index] = (1.-alphaaux)*qmatrix1value[sindex][s1index] + alphaaux*(payoffs1[s1index][s2index] + deltaQ*max(qmatrix1value[statespace.index(s)]))
                else:
                    alphaaux = 1.
                    qmatrix1value[sindex][s1index] = (1.-alphaaux)*qmatrix1value[sindex][s1index] + alphaaux*(payoffs1[s1index][s2index]+deltaQ*max(qmatrix1value[statespace.index(s)]))
                if q2count[sindex][s1index] > 1:
                    alphaaux = 1./(q2count[sindex][s2index]+1.)
                    qmatrix2value[sindex][s2index] = (1.-alphaaux)*qmatrix2value[sindex][s2index] + alphaaux*(payoffs2[s1index][s2index] + deltaQ*max(qmatrix2value[statespace.index(s)]))      
                else:
                    alphaaux = 1.
                    qmatrix2value[sindex][s2index] = (1.-alphaaux)*qmatrix2value[sindex][s2index] + alphaaux*(payoffs2[s1index][s2index]+ deltaQ*max(qmatrix2value[statespace.index(s)]))     
                if pricetraj1[t]==actionspace[0][1] and pricetraj2[t]==actionspace[1][1]:
                    collusioncount = collusioncount + 1./BatchsizeVector[nbatch-1]
                elif pricetraj1[t]==actionspace[0][2] and pricetraj2[t]==actionspace[1][2]:
                    collusioncount = collusioncount + 1./BatchsizeVector[nbatch-1]

            #Check if agents update strategy or not (based on the inertia parameter)
            inertiarandom1 = np.random.uniform(0.0, 1.0)
            inertiarandom2 = np.random.uniform(0.0, 1.0)
            if inertiarandom1 < 1. - inertia:
                #Post batch Q-value update for player 1
                for listitJ in range(M*M):
                    for listitK in range(M):
                        qmatrix1act[listitJ][listitK] = (1.-alpha)*qmatrix1act[listitJ][listitK] +alpha*qmatrix1value[listitJ][listitK]

            if inertiarandom2 < 1. - inertia:
                #Post batch Q-value update for player 2
                for listitJ in range(M*M):
                    for listitK in range(M):
                        qmatrix2act[listitJ][listitK] = (1.-alpha)*qmatrix2act[listitJ][listitK] +alpha*qmatrix2value[listitJ][listitK]


            #Calculate current strategy
            convergedstrat = np.zeros(2*M*M)
            for l in range(M*M):
                maxindex1 = np.argmax(qmatrix1act[l])
                maxindex2 = np.argmax(qmatrix2act[l])
                strategypairvec[l] =  maxindex1
                convergedstrat[l] = maxindex1
                strategypairvec[l+(M*M)] =  maxindex2
                convergedstrat[l+(M*M)] = maxindex2

            #Check if the strategy matches one of the absorbing states
            matchSymmetricNE = 1
            matchAllD = 1
            matchCollusiveNE = 0
            for listitF in range(M*M):
                if int(convergedstrat[listitF]) != int(AllDstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(AllDstrat[listitF+(M*M)]):
                    matchAllD = 0
            if int(convergedstrat[0]) != int(convergedstrat[9]) or int(convergedstrat[1]) != int(convergedstrat[12]) or int(convergedstrat[2]) != int(convergedstrat[15]) or int(convergedstrat[3]) != int(convergedstrat[10]) or int(convergedstrat[4]) != int(convergedstrat[13]) or int(convergedstrat[5]) != int(convergedstrat[16]) or int(convergedstrat[6]) != int(convergedstrat[11]) or int(convergedstrat[7]) != int(convergedstrat[14]) or int(convergedstrat[8]) != int(convergedstrat[17]):
                matchSymmetricNE = 0
            if matchSymmetricNE == 1 and int(convergedstrat[0]) == 1 and int(convergedstrat[4]) == 1:
                matchCollusiveNE = 1
            if matchSymmetricNE == 1 and int(convergedstrat[0]) == 2 and int(convergedstrat[8]) == 2:
                matchCollusiveNE = 1
            if matchSymmetricNE == 1 and int(convergedstrat[0]) == 1 and int(convergedstrat[4]) == 2 and int(convergedstrat[8]) == 2:
                matchCollusiveNE = 1

            #Record if it is one of the absorbing states in the appropriate list
            if matchSymmetricNE == 1:
                SymmetricNElist[int(nbatch)] = SymmetricNElist[int(nbatch)] + 1./NSamples   
            if matchCollusiveNE == 1:
                CollusiveNElist[int(nbatch)] =  CollusiveNElist[int(nbatch)] + 1./NSamples
            if matchAllD == 1:
                ADlist[int(nbatch)] =  ADlist[int(nbatch)] + 1./NSamples 

            Collist[int(nbatch)] = Collist[int(nbatch)] + collusioncount/NSamples


    return SymmetricNElist, CollusiveNElist, ADlist, Collist

#Run the simulation using numba
simulate_numba = njit(QvsQsimulation)
SymmetricNElist, CollusiveNElist, ADlist, Collist = simulate_numba(FinalT, qmatrix1initial, qmatrix2initial, actionspaceinitial, statespaceinitial, payoffs1, payoffs2, BatchsizeVector, deltaQ)

#Write the result of the simulation to text files.
for listitH in range(int(NBatches)):
    file1.write(str(ADlist[listitH]) + "\n")
    file2.write(str(SymmetricNElist[listitH]) + "\n")
    file3.write(str(CollusiveNElist[listitH]) + "\n")
    file4.write(str(Collist[listitH]) + "\n")

#Print how long the simulation took to run
progtime = time.time() - start_time
if progtime < 60:
    print("My program took", round(progtime, 2), "seconds to run.")
if progtime > 60:
    print("My program took", round(progtime/60, 2), "minutes to run.")

file1.close()
file2.close()
file3.close()
file4.close()
                   