# # 第一题
#
# def height(tree):
#     if tree == None:
#         return -1
#     else:
#         return 1 + max(height(tree.leftChild),height(tree.rightChild))
#
# def first(ctree):
#     if not ctree:
#         return True
#     else:
#         if abs(height(ctree.leftChild)-height(ctree.rightChild))>1:
#             return False
#         else:
#             return first(ctree.leftChild) and first(ctree.rightChild)
#
# ctree = BinaryTree(1)
# ctree.insertRight(2)
# ctree.insertLeft(3)
# ctree.getLeftChild().insertLeft(5)
# ctree.getLeftChild().getLeftChild().insertLeft(6)
# print(first(ctree))


# # 第二题
# def second(tree,value = 0):
#     if tree.isLeaf():
#         print(tree.getRootVal()*value)
#     else:
#         value += tree.getRootVal()
#         if tree.leftChild:
#             second(tree.getLeftChild(),value)
#         if tree.rightChild:
#             second(tree.getRightChild(), value)
#
#
#
#
# ctree = BinaryTree(1)
# ctree.insertRight(2)
# ctree.insertLeft(3)
# ctree.getLeftChild().insertLeft(5)
# ctree.getRightChild().insertLeft(5)
# ctree.getRightChild().insertRight(6)
# ctree.getLeftChild().getLeftChild().insertLeft(7)
# second(ctree)

# # 第三题
#
# def height(tree):
#     if tree == None:
#         return 0
#     else:
#         return 1 + max(height(tree.leftChild), height(tree.rightChild))
#
#
# def third(tree,zhijing=[]):
#     if tree.isLeaf():
#         return 0
#     else:
#         if tree.rightChild:
#             right_height = height(tree.getRightChild())
#             print(right_height)
#             third(tree.getRightChild())
#         else:
#             right_height = 0
#         if tree.leftChild:
#             left_height = height(tree.getLeftChild())
#             third(tree.getLeftChild())
#         else:
#             left_height = 0
#     zhijing.append(left_height+right_height)
#     return zhijing
#
#
#
# ctree = BinaryTree(1)
# ctree.insertRight(2)
# ctree.insertLeft(3)
# ctree.getLeftChild().insertLeft(5)
# ctree.getRightChild().insertLeft(8)
# ctree.getRightChild().insertRight(6)
# ctree.getLeftChild().getLeftChild().insertLeft(7)
# print(max(third(ctree)))


# # 第四题
# def fourth(tree, maxv=0):
#     if tree.isLeaf():
#         return tree.getRootVal() ,max(maxv,tree.getRootVal())
#     else:
#         if tree.rightChild and tree.leftChild:
#             leftsum ,maxleft =  fourth(tree.getLeftChild(), maxv)
#             rightsum,maxright = fourth(tree.getRightChild(), maxv)
#
#             sumv = leftsum+rightsum+tree.getRootVal()
#
#             maxv = max(sumv,maxv,maxleft,maxright)
#             return sumv,maxv
#         elif tree.rightChild and not tree.leftChild:
#             rightsum, maxright = fourth(tree.getRightChild(), maxv)
#
#             sumv = rightsum+tree.getRootVal()
#             maxv = max(sumv, maxv,maxright)
#             return sumv,maxv
#         elif not tree.rightChild and tree.leftChild:
#             leftsum, maxleft = fourth(tree.getLeftChild(), maxv)
#
#
#             sumv = leftsum + tree.getRootVal()
#             maxv = max(sumv, maxv,maxleft)
#
#             return sumv,maxv
#
# ctree = BinaryTree(-4)
# ctree.insertRight(3)
# ctree.insertLeft(-2)
# ctree.getLeftChild().insertLeft(11)
# ctree.getRightChild().insertLeft(2)
# ctree.getRightChild().insertRight(-5)
# ctree.getRightChild().getRightChild().insertRight(4)
# ctree.getLeftChild().getLeftChild().insertLeft(-3)
# ctree.getLeftChild().getLeftChild().insertRight(1)
#
# print(fourth(ctree)[1])


# # 第五题
#
# from heapq import *    # 利用python的heapq库,来实现堆
# class fifth:
#     def __init__(self):
#         self.small = []
#         self.large = []
#     def Insert(self, num):
#         # write code here
#         small,large = self.small,self.large
#         heappush(small,-heappushpop(large,-num))
#         if len(large) < len(small):
#         # 这样就保持了大顶堆和小顶堆元素个数的相对稳定性，在取中位数的时候也方便。
#             heappush(large,-heappop(small))
#     def GetMedian(self,n=1):
#         # write code here
#         small,large = self.small,self.large
#         if len(large) > len(small):
#             return float(-large[0])
#         return (small[0] - large[0]) / 2.0


# # # 第六题
# class unionfind:
#     def __init__(self, groups):
#         self.groups = groups
#         self.items = []
#         for g in groups:
#             self.items += list(g)
#         self.items = set(self.items)
#
#         self.parent = {}
#         self.rootdict = {}  # 记住每个root下节点的数量
#         for item in self.items:
#             self.rootdict[item] = 1
#             self.parent[item] = item
#
#     def union(self, r1, r2):
#         rr1 = self.findroot(r1)
#         rr2 = self.findroot(r2)
#         cr1 = self.rootdict[rr1]
#         cr2 = self.rootdict[rr2]
#         if cr1 >= cr2:  # 将节点数量较小的树归并给节点数更大的树
#             self.parent[rr2] = rr1
#             self.rootdict.pop(rr2)
#             self.rootdict[rr1] = cr1 + cr2
#         else:
#             self.parent[rr1] = rr2
#             self.rootdict.pop(rr1)
#             self.rootdict[rr2] = cr1 + cr2
#
#     def findroot(self, r):
#         """
#         可以通过压缩路径来优化算法,即遍历路径上的每个节点直接指向根节点
#         """
#         if r in self.rootdict.keys():
#             return r
#         else:
#             return self.findroot(self.parent[r])
#
#     def createtree(self):
#         for g in self.groups:
#             if len(g) < 2:
#                 continue
#             else:
#                 for i in range(0, len(g) - 1):
#                     if self.findroot(g[i]) != self.findroot(g[i + 1]):  # 如果处于同一个集合的节点有不同的根节点，归并之
#                         self.union(g[i], g[i + 1])
#
#     def printree(self):
#         rs = {}
#         for item in self.items:
#             root = self.findroot(item)
#             rs.setdefault(root, [])
#             rs[root] += [item]
#         temp_list = []
#         for key in rs.keys():
#             temp_list.append(rs[key])
#         return temp_list
# m = [[0,1,0,0,1,1,0,1],
#      [0,0,0,0,0,0,1,1],
#      [1,1,0,1,0,1,0,1],
#      [0,0,1,1,0,1,1,1],
#      [1,0,0,1,0,1,0,0],
#      [1,0,1,0,0,0,0,0],
#      [1,0,1,0,0,0,1,0],
#      [0,0,0,1,1,0,0,1]]
# temp_list = []
# for i in range(len(m)):
#     temp = []
#     for j in range(len(m)):
#         temp.append(m[j][i])
#     temp_list.append(tuple(temp))
# m+=temp_list
#
# connect_list = []
# for i in range(len(m)):
#     if i <len(m[i]):
#         for j in range(len(m[i])):
#             temp_con=[]
#             if m[i][j] ==1:
#                 temp_con.append((i)*len(m[i])+j+1)
#                 if i != 0 and i!= len(m[i])-1 and j != 0 and j != len(m[i])-1:
#                     if m[i+1][j] ==1:
#                         temp_con.append((i+1)*len(m[i])+j+1)
#                     if m[i-1][j] ==1:
#                         temp_con.append((i-1)*len(m[i])+j+1)
#                     if m[i][j+1] ==1:
#                         temp_con.append((i)*len(m[i])+j+2)
#                     if m[i][j-1] ==1:
#                         temp_con.append((i)*len(m[i])+j)
#                 if i == 0 and i!= len(m[i])-1 and j != 0 and j != len(m[i])-1:
#                     if m[i+1][j] ==1:
#                         temp_con.append((i+1)*len(m[i])+j+1)
#                     if m[i][j+1] ==1:
#                         temp_con.append((i)*len(m[i])+j+2)
#                     if m[i][j-1] ==1:
#                         temp_con.append((i)*len(m[i])+j)
#
#                 if i != 0 and i== len(m[i])-1 and j != 0 and j != len(m[i])-1:
#                     if m[i-1][j] ==1:
#                         temp_con.append((i-1)*len(m[i])+j+1)
#                     if m[i][j+1] ==1:
#                         temp_con.append((i)*len(m[i])+j+2)
#                     if m[i][j-1] ==1:
#                         temp_con.append((i)*len(m[i])+j)
#                 if i != 0 and i!= len(m[i])-1 and j == 0 and j != len(m[i])-1:
#                     if m[i+1][j] ==1:
#                         temp_con.append((i+1)*len(m[i])+j+1)
#                     if m[i-1][j] ==1:
#                         temp_con.append((i-1)*len(m[i])+j+1)
#                     if m[i][j+1] ==1:
#                         temp_con.append((i)*len(m[i])+j+2)
#                 if i != 0 and i!= len(m[i])-1 and j != 0 and j == len(m[i])-1:
#                     if m[i+1][j] ==1:
#                         temp_con.append((i+1)*len(m[i])+j+1)
#                     if m[i-1][j] ==1:
#                         temp_con.append((i-1)*len(m[i])+j+1)
#                     if m[i][j-1] ==1:
#                         temp_con.append((i)*len(m[i])+j)
#             connect_list.append(tuple(temp_con))
#
# def final_cal(list,m):
#     for i in list:
#         for j in i:
#             if j % 8 ==0:
#                 a,b = (j//8)-1 ,8
#             else:
#                 a,b = j//8,j%8
#             print(a,b)
#             m[a][b-1] = len(i)
#     return m
#
# u = unionfind(connect_list)
# u.createtree()
# temp_res = u.printree()
# print(temp_res)
# res = final_cal(temp_res,m[:8])
# print(res)

# # 图第一题
# bellman-ford
# G = {0:{1:1,5:1},
#      1: {0: 1, 2: 1, 6: 1},
#      2: {1: 1, 3: 1},
#      3: {2: 1, 6: 1,4:1},
#      4: {3: 1, 5: 1,6:1},
#      5: {4: 1,0:1},
#      6:{1:1,3:1,4:1}}
#
# def getEdges(G):
#     """ 读入图G，返回其边与端点的列表 """
#     v1 = []     # 出发点
#     v2 = []     # 对应的相邻到达点
#     w  = []     # 顶点v1到顶点v2的边的权值
#     for i in G:
#      for j in G[i]:
#          if G[i][j] != 0:
#              w.append(G[i][j])
#              v1.append(i)
#              v2.append(j)
#     return v1,v2,w
#
# def Bellman_Ford(G, v0, INF=999):
#      v1,v2,w = getEdges(G)
#
#      # 初始化源点与所有点之间的最短距离
#      dis = dict((k,INF) for k in G.keys())
#      dis[v0] = 0
#
#      # 核心算法
#      for k in range(len(G)-1):   # 循环 n-1轮
#          check = 0           # 用于标记本轮松弛中dis是否发生更新
#          for i in range(len(w)):     # 对每条边进行一次松弛操作
#              if dis[v1[i]] + w[i] < dis[v2[i]]:
#                  dis[v2[i]] = dis[v1[i]] + w[i]
#                  check = 1
#          if check == 0: break
#
#      # 检测负权回路
#      # 如果在 n-1 次松弛之后，最短路径依然发生变化，则该图必然存在负权回路
#      flag = 0
#      for i in range(len(w)):             # 对每条边再尝试进行一次松弛操作
#         if dis[v1[i]] + w[i] < dis[v2[i]]:
#             flag = 1
#             break
#         if flag == 1:
#         #         raise CycleError()
#             return False
#         return dis
#
# v0 = 0
# dis = Bellman_Ford(G, v0)
# print(dis.values())


# # 图第二题
#
#
# def topological_sort(graph):
#     is_visit = dict((node, False) for node in graph)
#     li = []
#
#     def dfs(graph, start_node):
#
#         for end_node in graph[start_node]:
#             if not is_visit[end_node]:
#                 is_visit[end_node] = True
#                 dfs(graph, end_node)
#         li.append(start_node)
#
#     for start_node in graph:
#         if not is_visit[start_node]:
#             is_visit[start_node] = True
#             dfs(graph, start_node)
#
#     li.reverse()
#     return li
#
#
# if __name__ == '__main__':
#     g= {3:[1,6],
#         1:[2,4],
#         2:[5],
#         4:[2,6],
#         5:[],
#         6:[5]}
#     li = topological_sort(g)
#     print(li)


# # 图第三题
#
#
# def findAllPath(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return [path]
#
#     paths = []  # 存储所有路径
#     for node in graph[start]:
#         if node not in path:
#             newpaths = findAllPath(graph, node, end, path)
#             for newpath in newpaths:
#                 paths.append(newpath)
#     return paths
#
#
#
#
# g= {0:{1,2,3},
#     1:{4},
#     2:{4,5},
#     3:{5},
#     4:{6,7},
#     5:{7},
#     6:{8},
#     7:{8},
#     8:{}}
# print(findAllPath(g,0,8))


# # 图第四题
#
#
# def isBipartite(graph):
#     n = len(graph)
#     record = [0] * n
#
#     def dfs(point, c):
#         record[point] = c
#         for i in graph[point]:
#             # 访问为-c的点直接跳过
#             if record[i] == -c:
#                 continue
#             # 染色相同则不能为二分图
#             elif record[i] == c:
#                 return False
#             # 对于没访问过的点染色
#             elif record[i] == 0 and not dfs(i, -c):
#                 return False
#         return True
#
#     for i in range(n):
#         # 图不是连通的
#         if record[i] == 0 and not dfs(i, 1):
#             return False
#     return True
#
#
# graph = {0: {1, 2},
#          1: {0, 3},
#          2: {0, 3},
#          3: {1, 2}}
# print(isBipartite(graph))



# # 图第五题
#
# def getAdjMatrix():
#     edge = [[0,1,1,1],
#             [1,0,1,1],
#             [1,1,0,1],
#             [1,1,1,0]]
#     pointNum = 4
#     return edge , pointNum
#
#
#
# def main():
#     edge, pointNum = getAdjMatrix()
#     print('')
#     for i in edge:
#         print('    ', end='')
#         for j in i:
#             print(j, end='\t\t')
#         print('\n')
#     colorNum = 0
#     disabled = []
#
#     # 初始化color列表，用以记录每个顶点的着色情况
#     color = []
#     for i in range(pointNum):
#         color.append(0)
#     edgeNum = [sum(e) for e in edge]
#     for k in range(pointNum):
#         # 获取顶点最大度的索引值
#         maxEdgePoint = [i for i in range(pointNum) if edgeNum[i] == max(edgeNum) and edgeNum[i] != 0]
#         # 遍历最大度
#         for p in maxEdgePoint:
#             if p not in disabled:
#                 # 选取还未着色且度最大的点p开始着色
#                 color[p] = colorNum + 1
#                 disabled.append(p)
#                 edgeNum[p] = 0
#                 # temp用于查找该颜色可用来着色的下一个顶点
#                 temp = edge[p]
#                 for i in range(pointNum):
#                     if i not in disabled:
#                         if temp[i] == 0:
#                             # 为不冲突的顶点着色
#                             color[i] = colorNum + 1
#                             disabled.append(i)
#                             edgeNum[i] = 0
#                             # 增加当前颜色的禁忌点
#                             temp = [x + y for (x, y) in zip(edge[i], temp)]
#                 # 需要新颜色
#                 colorNum = colorNum + 1
#
#         # 每个顶点都已经着色
#         if 0 not in color:
#             break
#     print(color)
#     print(colorNum)
#
#
# main()


# # 图第六题
#
# import numpy as np
#
# def link_num(graph,k):
#     graph1 = graph
#     graph2 = graph1
#     if k ==1:
#         return graph1
#     else:
#         for i in range(k-1):
#             graph3 = graph1@graph2
#             graph2 = graph3
#         return graph3
#
# graph = np.array([[0,1,0,1],
#                   [0,0,1,1],
#                   [1,1,0,1],
#                   [0,0,1,0]])
# res_mat = link_num(graph,2)
# print(res_mat)
# res = sum(sum(res_mat))
# print(res)

