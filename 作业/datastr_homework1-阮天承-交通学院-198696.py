# 1.由于最坏情况的算法复杂度为O（n），最好情况的算法复杂度为O（1），平均时间复杂度为O（n/2）
# 由于O（n/2）=O（n）,所以该算法的平均情况下的时间复杂度为O（n）

# 2.计算鸭子
# def duckcount(n,x):
#     if n==0:
#         return 2
#
#     a=(x+2)*2
#     x=a
#     duckcount(n-1,x)
#     if n == 1:
#         print(a)
# duckcount(5,2)

# # 3.列表原地逆置
# def list_reserve(A):
#
#     for i in range(len(A)):#检验是否有下一层列表
#         if type(A[i])==list:
#             A[i] = list_reserve(A[i])#下层列表递归
#     return list(reversed(A))  # 逆置
#
# A = [1, [2, 3], 4, [5, [6, 7], 8], 9]
# A = list_reserve(A)
# print(A)

# 4.最长公共子序列Ics
# import numpy
# def find_lcseque(s1, s2):
#     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
#     m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
#     # d用来记录转移方向
#     d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
#
#     for p1 in range(len(s1)):
#         for p2 in range(len(s2)):
#             if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
#                 m[p1 + 1][p2 + 1] = m[p1][p2] + 1
#                 d[p1 + 1][p2 + 1] = 'ok'
#             elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
#                 m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
#                 d[p1 + 1][p2 + 1] = 'left'
#             else:  # 上值大于左值，则该位置的值为上值，并标记方向up
#                 m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
#                 d[p1 + 1][p2 + 1] = 'up'
#     (p1, p2) = (len(s1), len(s2))
#     # print(numpy.array(d))
#     s = []
#     while m[p1][p2]:  # 不为None时
#         c = d[p1][p2]
#         if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
#             s.append(s1[p1 - 1])
#             p1 -= 1
#             p2 -= 1
#         if c == 'left':  # 根据标记，向左找下一个
#             p2 -= 1
#         if c == 'up':  # 根据标记，向上找下一个
#             p1 -= 1
#     s.reverse()
#     return ''.join(s)
#
# A = 'abdebcbb'
# B = 'adacbcb'
# print(find_lcseque(A, B))

#
# # # 5.全排列(没做完)
# import copy
# def all_list(S,n,k,list=''):
#     if k ==0:
#         print(list)
#
#     else:
#         for i in range(len(S[2-k])):
#             list += str(S[2-k][i])
#             all_list(S,n,k-1,list)
#             list = list[:-1]
#
#
# A = [1,2,3]
# n = 3
# k = 2
# S= []
# for i in range(k):
#     S.append(copy.deepcopy(A))
# all_list(S,n,k)


# # 6.和的子集合

# from itertools import combinations
#
# def num(S,n, K):
#
#     res = []
#     for i in range(len(S)):
#         res += list(combinations(S, i))
#     res = [x for x in res if len(x) == n]
#     a = []
#     for j in res:
#         if sum(j) == K:
#             a.append(list(j))
#     return a
# S = [4, 5, 7, 3, 9, 6, 2]
# K = 15
# for i in range(len(S)):
#
#     res = num(S,i, K)
#     if len(res):
#         print(res)


# # 7.等和子集合
# def origin_operation(S, m, sub_list=[]):
#     sum_S = sum(S)
#     aim = sum_S / m
#     S.sort(reverse=True)
#     print(S)
#     for i in range(m):
#         if i != m - 1:
#             sub_list.append([S[i]])
#         else:
#             sub_list.append(S[m - 1:])
#     return aim, sub_list
#
# def sum_sub(aim, sub_list, rest=[123], pop_n=[]):
#     if min(rest) == 0:
#         print(sub_list)
#         return 0
#     sum_l = []
#     for i in range(len(sub_list)):
#         sum_l.append(sum(sub_list[i]))
#     rest = [i - aim for i in sum_l]
#     error_low = abs(min(rest))
#     error_high = max(rest)
#     max_n = sum_l.index(max(sum_l))
#     min_n = sum_l.index(min(sum_l))
#     maxl = sub_list[max_n]
#     minl = sub_list[min_n]
#     maxl_con = [i for i in maxl]
#     # maxl_con = [i for i in maxl if i <= error_high and i <= error_low]
#     min_maxl_con = maxl_con.pop(maxl_con.index(min(maxl_con)))
#     # if pop_n.count(min_maxl_con) >= 1:
#     #     min_maxl_con = maxl_con.pop(maxl_con.index(max(maxl_con)))
#     # if len(pop_n) <= 2:
#     #     pop_n.append(min_maxl_con)
#     # else:
#     #     pop_n[0] = min_maxl_con
#
#     minl.append(min_maxl_con)
#     maxl.remove(min_maxl_con)
#     # minl.append(maxl.pop(maxl.index(max(maxl_con))))
#     sub_list[max_n] = maxl
#     sub_list[min_n] = minl
#     print(sub_list, rest, pop_n)
#     sum_sub(aim, sub_list, rest)
#
#
# S = [6, 8, 7, 7, 4, 4, 6, 3, 1, 2]
# m = 3
# aim, sub_list = origin_operation(S, m)
# sum_sub(aim, sub_list)

# # 8.素数环
# def Perm(a, k, n,res=[]):
#     # n 是数组a的元素个数，生成a[k],…,a[n-1]的全排列
#     if k == n - 1:  # 终止条件，输出排列
#         res.append(a[:n])  # 输出包括前缀，以构成整个问题的解
#     else:  # a[k],…,a[n-1] 的排列大于1，递归生成
#         for i in range(k, n):
#             a[k], a[i] = a[i], a[k]  # 交换a[k]和 a[i]
#             Perm(a, k + 1, n)  # 生成 a[k+1],…,a[n-1]的全排列
#             a[k], a[i] = a[i], a[k]  # 再次交换 a[k] 和a[i] , 恢复原顺序
#     return res
#
# def sushu(res,su):
#     for i in res:
#
#         enable = True
#         for j in range(len(i)-1):
#             # print(j)
#             if not i[j]+i[j+1] in su:
#                 enable = False
#                 break
#         if enable:
#             print(i)
#
#
# a=[1,2,3,4,5,6,7,8,9,10]
# k=0
# n=10
# res=Perm(a,k,n)
#
# su = [2,3,5,7,11,13,17,19,23,29]
# sushu(res,su)
