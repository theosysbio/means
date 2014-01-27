import itertools
def fcount(n_moments,n_vars):

    """
    :param n_moments: the maximal order of moment to be computer
    :param n_vars: the number of variables
    :return: a pair of tuples. the first element contains the all the permutations,
    whilst the second element does not have the first order (e.g. {0,0,1})
    """

    #todo we should really make a list of tuples here, but we kept list of list otherwise test fail...
    m_counter = [list(i) for i in itertools.product(range(0, n_moments + 1), repeat = n_vars) if sum(i) <= n_moments]
    counter = [i for i in m_counter if sum(i) != 1]
    return (counter, m_counter)




###############################################################
# Countiterate function creates list of possible moment combinations
# that sum to < or = nMoments
# T = nMoments, N = nvariables, nvec = list of zeros of length N
# 'counter' is a list of required nvec combinations
# For initial call, VEC = -1 (this will be updated as it is called
# recursively)
################################################################
#
# def countiterates(VEC,T,N,counter,nvec):
#     for i in range(0,T+1):
#         nvec[len(nvec)-N] = i
#         if N == 1:
#             Dnumber=sum(nvec)
#             if Dnumber<(T+1):
#                 VEC+=1
#                 counter[VEC]=nvec[:]
#         else:
#             [VEC, counter, nvec] = countiterates(VEC,T,N-1,counter,nvec)
#     return [VEC, counter, nvec]
#
# #
# # ###################################################################
# # # fcount formats the output received from countiterates, to return
# # # 'counter' and 'mcounter' which are used in subsequent calculations.
# # # 'counter' doesn't include 1st order moments, 'mcounter' does
# # ###################################################################
# #
# def fcount(nMoments,nvariables):
#
#     counter = [[0]*nvariables] * ((nMoments+1)**nvariables)    #create empty counter
#     nvec = [0] * nvariables
#
#     #countiterates(0,nMoments,nvariables,counter,nvec) to get filled counter
#
#     [VEC, counter, nvec] = countiterates(-1,nMoments,nvariables,counter,nvec)
#
#     countersum = []                   #create list where entry i is sum of ith row in counter
#     for i in range(0,len(counter)):
#         row = counter[i]
#         countersum.append(sum(row))
#     idx = []                          #create a list of the indices of the nonzero entries in countersum
#     for i in range(0,len(counter)):
#         if countersum[i]!=0:
#             idx.append(i)
#     nTerms = max(idx)+1
#     mcounter = []                     #mcounter is the list of all the entries in counter up to i=max(idx)
#     for i in range(0,nTerms):
#         mcounter.append(counter[i])
#     counter = mcounter
#     Mcounter = []
#     for i in range(0,nTerms):
#         row = [countersum[i],mcounter[i]]
#         Mcounter.append(row)
#     Mcounter.sort()                   #sort Mcounter to order mcounter
#     mcounter = []
#     for i in range(0,nTerms):
#         row = Mcounter[i]
#         row = row[1]
#         mcounter.append(row)
#     mcountersum = []
#     for i in range(0,len(mcounter)):
#         row = mcounter[i]
#         mcountersum.append(sum(row))
#     counter = []
#     for i in range(0,len(mcounter)):
#         row = mcounter[i]
#         if sum(row)!=1:               #ignores entries which sum to 1, ie first order moments
#             counter.append(row)
#
#     return [counter, mcounter]
