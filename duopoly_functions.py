import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import fsolve

#Environment functions
def FOC(prices, *modelpara):
        p1, p2 = prices
        a1, a2, mu, c1, c2 = modelpara
        return ((mu*math.exp((2*(a1 + p2))/mu) + 
 math.exp((a1 + p1 + p2)/mu)*(math.exp(a2/mu) + math.exp(p2/mu))*(mu + c1 - p1))/(mu*(math.exp((a2 + p1)/mu) + math.exp((a1 + p2)/mu) + math.exp((p1 + p2)/mu))**2), (mu*math.exp((2*(a2 + p1))/mu) + 
 math.exp((a2 + p1 + p2)/mu)*(math.exp(a1/mu) + math.exp(p1/mu))*(mu + c2 - p2))/(mu*(math.exp((a2 + p1)/mu) + math.exp((a1 + p2)/mu) + math.exp((p1 + p2)/mu))**2))

def FOCJoint(prices, *modelpara):
    p1, p2 = prices
    a1, a2, mu, c1, c2 = modelpara
    return (math.exp((a1 + p2)/mu)*(math.exp(p2/mu)*(mu*math.exp(a1/mu) + math.exp(p1/mu)*(mu + c1 - p1)) + 
   math.exp((a2 + p1)/mu)*(mu + c1 - c2 - p1 + p2))/(mu*(math.exp((a2 + p1)/mu) + math.exp((a1 + p2)/mu) + math.exp((p1 + p2)/mu))**2), math.exp((a2 + p1)/mu)*(math.exp(p1/mu)*(mu*math.exp(a2/mu) + math.exp(p2/mu)*(mu + c2 - p2)) + 
   math.exp((a1 + p2)/mu)*(mu + c1 - c2 - p1 + p2))/(mu*(math.exp((a2 + p1)/mu) + math.exp((a1 + p2)/mu) + math.exp((p1 + p2)/mu))**2))
    
def FOCsim(a, b, *modelpara):
    return fsolve(FOC, [a, b], args=modelpara)

def FOCsimJoint(a, b, *modelpara):
    return fsolve(FOCJoint, [a, b], args=modelpara)

def priceintervalgen(option, xi, nashprices, monprices):
    if option == 'Calvano':
        priceintervalinit = [[nashprices[0] - xi*(monprices[0]-nashprices[0]), monprices[0]+xi*(monprices[0]-nashprices[0])], [nashprices[1] - xi*(monprices[1]-nashprices[1]), monprices[1]+xi*(monprices[1]-nashprices[1])]]
        return priceintervalinit
    if option == 'SymmetricNash':
        priceintervalinit = [[nashprices[0] - (xi+1)*(monprices[0]-nashprices[0]), nashprices[0] - (xi+1)*(monprices[0]-nashprices[0])+2*(xi+1)*(monprices[0]-nashprices[0])], [nashprices[1] - (xi+1)*(monprices[1]-nashprices[1]), nashprices[1] - (xi+1)*(monprices[1]-nashprices[1])+2*(xi+1)*(monprices[1]-nashprices[1])]]
        return priceintervalinit

def actionspacegen(option, M, xi, nashprices, monprices):
    actionspaceinit=[[], []]
    priceinterval = priceintervalgen(option, xi, nashprices, monprices)
    for i in range(M):
        actionspaceinit[0].append(priceinterval[0][0] + i*(priceinterval[0][1]-priceinterval[0][0])/(M-1) ) 
        actionspaceinit[1].append(priceinterval[1][0] + i*(priceinterval[1][1]-priceinterval[1][0])/(M-1) )    
    return actionspaceinit

def statespacegen(M, actionspace):
    statespaceinit=[]
    for i in range(M):
        for j in range(M):
            statespaceinit.append([actionspace[0][i], actionspace[1][j]])
    return statespaceinit
   
def reward(s, i, actionspace, a1, a2, mu, c1, c2):
    demand = [math.exp((a1-actionspace[0][s[0]])/mu)/(math.exp((a1-actionspace[0][s[0]])/mu)+math.exp((a2-actionspace[1][s[1]])/mu) + 1), math.exp((a2-actionspace[1][s[1]])/mu)/(math.exp((a1-actionspace[0][s[0]])/mu)+math.exp((a2-actionspace[1][s[1]])/mu) + 1.)]
    reward = [demand[0]*(actionspace[0][s[0]]-c1), demand[1]*(actionspace[1][s[1]]-c2)] 
    return reward[i]

def payoffmatrix(m, M, actionspace, a1, a2, mu, c1, c2):
    payoffmat = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            if m == 0:
                payoffmat[i][j] = float(reward([i, j], m, actionspace, a1, a2, mu, c1, c2))
            if m == 1:
                payoffmat[i][j] = float(reward([i, j], m, actionspace, a1, a2, mu, c1, c2))
    return payoffmat

def effectiveNash(m, M, actionspace, a1, a2, mu, c1, c2):
    if m == 0:
        Tpayoffs = payoffmatrix(m, M, actionspace, a1, a2, mu, c1, c2).transpose()
    else:
        Tpayoffs = payoffmatrix(m, M, actionspace, a1, a2, mu, c1, c2)
    for i in range(M):
        maxpay = max(Tpayoffs[i])
        maxindex = np.where(Tpayoffs[i] == maxpay)[0][0]
        if maxindex == i:
            effectivenashrev = Tpayoffs[i][i]
            nashindex = i
    return [effectivenashrev, actionspace[m][nashindex], nashindex]

def effectiveMon(m, M, actionspace, a1, a2, mu, c1, c2):
    effectiveMonrev = max(payoffmatrix(m, M, actionspace, a1, a2, mu, c1, c2).diagonal())
    monindex = np.where(payoffmatrix(m,M, actionspace, a1, a2, mu, c1, c2).diagonal() == effectiveMonrev)[0][0]
    return [effectiveMonrev, actionspace[m][monindex], monindex]

def nashRevenue(i, nashprices, a1, a2, mu, c1, c2):
    nashdemand = [math.exp((a1-nashprices[0])/mu)/(math.exp((a1-nashprices[0])/mu)+math.exp((a2-nashprices[1])/mu) + 1), math.exp((a2-nashprices[1])/mu)/(math.exp((a1-nashprices[0])/mu)+math.exp((a2-nashprices[1])/mu) + 1.)]
    nashreward = [nashdemand[0]*(nashprices[0]-c1), nashdemand[1]*(nashprices[0]-c2)] 
    return nashreward[i]

def monopolyRevenue(i, monprices, a1, a2, mu, c1, c2):
    mondemand = [math.exp((a1-monprices[0])/mu)/(math.exp((a1-monprices[0])/mu)+math.exp((a2-monprices[1])/mu) + 1), math.exp((a2-monprices[1])/mu)/(math.exp((a1-monprices[0])/mu)+math.exp((a2-monprices[1])/mu) + 1.)]
    monreward = [mondemand[0]*(monprices[0]-c1), mondemand[1]*(monprices[0]-c2)] 
    return monreward[i]

def ConsumerSurp(price1, price2, a1, a2, mu, c1, c2):
    return (-1)*math.log(1-math.exp((a1-price1)/mu)/(math.exp((a1-price1)/mu)+math.exp((a2-price2)/mu) + 1)-math.exp((a1-price2)/mu)/(math.exp((a1-price1)/mu)+math.exp((a2-price2)/mu) + 1))

def nashConsumerSurp(i, nashprices, a1, a2, mu, c1, c2):
    return (-1)*math.log(1-2*math.exp((a1-nashprices[i])/mu)/(math.exp((a1-nashprices[0])/mu)+math.exp((a2-nashprices[1])/mu) + 1))

def monConsumerSurp(i, monprices, a1, a2, mu, c1, c2):
    return (-1)*math.log(1-2*math.exp((a1-monprices[i])/mu)/(math.exp((a1-monprices[0])/mu)+math.exp((a2-monprices[1])/mu) + 1))

#Q-learning functions

def IncreasingBatchK(InitialBatchK, lincons, count):
    return math.ceil(InitialBatchK + lincons*count)

def FinalTime(InitialBatchK, lincons, NBatches):
    timesum = 0
    for k in range(NBatches):
        timesum = timesum + math.ceil(InitialBatchK + lincons*k)
    return timesum

def epsilon(t, beta):
    return math.exp(-t*beta)

def Qinitialization(option, M, m, delta, payoffs):
    if option == 'std_normal':
        return np.random.normal(0, 1, size=(M*M, M))
    elif option == 'uniform':
        return np.random.uniform(0, 1, size=(M*M, M))
    elif option == 'constant':
        return np.full((M*M,M), 0.5)
    elif option == 'constant_scaled':
        return np.full((M*M,M), 1./(1.-delta))
    elif option == 'Calvano':
        qmatrixtemp = np.zeros((M*M, M))
        if m == 0:
            for i in range(M):
                aveactionpay = 0
                for j in range(M):
                    aveactionpay += payoffs[i][j]
                for l in range(M*M):
                    qmatrixtemp[l][i] = aveactionpay/((1-delta)*(M)) 
            return qmatrixtemp
        if m == 1:
            for i in range(M):
                aveactionpay = 0
                for j in range(M):
                    aveactionpay += payoffs[j][i]
                for l in range(M*M):
                    qmatrixtemp[l][i] = aveactionpay/((1-delta)*(M))      
            return qmatrixtemp
    else:
        raise ValueError("Invalid initialization option.")

#Exp3 functions    

def Exp3StratDist(Exp3, M, eta):
    dist = [0]*M
    normalize = 0.
    for l in range(M):
        normalize = normalize + math.exp(eta*Exp3[l])
    for m in range(M):
        dist[m] = math.exp(eta*Exp3[m])/normalize
    return dist

def optimal_eta(M, deltaExp):
    return math.sqrt(math.log(float(M))*(1-deltaExp**2)/(float(M)*(deltaExp**2)))
    

#Evaluation functions

def strategypair2D(qmatrix1, qmatrix2):
    if qmatrix1[0][0]>qmatrix1[0][1] and qmatrix2[0][0]>qmatrix2[0][1] and qmatrix1[1][0]>qmatrix1[1][1] and qmatrix2[1][0]>qmatrix2[1][1] and qmatrix1[2][0]>qmatrix1[2][1] and qmatrix2[2][0]>qmatrix2[2][1] and qmatrix1[3][0]>qmatrix1[3][1] and qmatrix2[3][0]>qmatrix2[3][1]:
        return 0
    if qmatrix1[0][0]>qmatrix1[0][1] and qmatrix2[0][0]>qmatrix2[0][1] and qmatrix1[1][0]>qmatrix1[1][1] and qmatrix2[1][0]>qmatrix2[1][1] and qmatrix1[2][0]>qmatrix1[2][1] and qmatrix2[2][0]>qmatrix2[2][1] and qmatrix1[3][0]<qmatrix1[3][1] and qmatrix2[3][0]<qmatrix2[3][1]:
        return 1
    if qmatrix1[0][0]<qmatrix1[0][1] and qmatrix2[0][0]<qmatrix2[0][1] and qmatrix1[1][0]>qmatrix1[1][1] and qmatrix2[1][0]>qmatrix2[1][1] and qmatrix1[2][0]>qmatrix1[2][1] and qmatrix2[2][0]>qmatrix2[2][1] and qmatrix1[3][0]<qmatrix1[3][1] and qmatrix2[3][0]<qmatrix2[3][1]:
        return 2
    else:
        return 3

def strategypairGeneral(M, qmatrix1, qmatrix2):
    strategypairvec=[0]*(2*M*M)
    for n in range(M*M):
        maxindex1 = np.where(qmatrix1[n] == max(qmatrix1[n]))
        maxindex2 = np.where(qmatrix2[n] == max(qmatrix2[n]))
        strategypairvec[n] =  maxindex1[0][0]
        strategypairvec[n+(M*M)] =  maxindex2[0][0]
    return strategypairvec


def strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace):
    graph = [0]*(M*M)
    absorbstate = []
    for n in range(M*M):
        #maxindex1 = np.where(qmatrix1[n] == max(qmatrix1[n]))
        #maxindex2 = np.where(qmatrix2[n] == max(qmatrix2[n]))
        maxindex1 = np.argmax(qmatrix1[n])
        maxindex2 = np.argmax(qmatrix2[n])
        graph[n] =  statespace.index([actionspace[0][maxindex1], actionspace[1][maxindex2]])   
        if graph[n] == n:
            absorbstate.append(n)
    return absorbstate

def nashabsorbstate(M, statespace, actionspace, a1, a2, mu, c1, c2):
    nprice1temp = effectiveNash(0, M, actionspace, a1, a2, mu, c1, c2)[1]
    nprice2temp = effectiveNash(1, M, actionspace, a1, a2, mu, c1, c2)[1]
    return statespace.index([nprice1temp, nprice2temp])

def collusivestratpair(M, qmatrix1, qmatrix2, nashstate, actionspace, statespace):
    if len(strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace))>0 and strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace)[0]>nashstate:
        return 2
    if len(strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace))>0 and strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace)[0]==nashstate:
        return 1
    else:
        return 0

