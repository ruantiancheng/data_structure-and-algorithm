#!/usr/bin/env python
# -*- coding:utf-8 -*-


# # 1.多重背包问题
# import numpy as np
# def multi_tran_01(weightList,valueList,numList):
#     '''
#     转换函数，将多重背包转换成0-1背包
#     :param weightList: 物品重量
#     :param valueList: 物品价值
#     :param numList: 物品数量
#     :return: 0-1环境下的物品属性数组
#     '''
#     K = len(numList) #物品类数
#     newWeightList = []
#     newValueList = []
#     num = sum(numList) #获取物品总数
#     for i in range(K):
#         for j in range(numList[i]):
#             newWeightList.append(weightList[i])
#             newValueList.append(valueList[i])
#     return num,newValueList,newWeightList
#
# def choice(choice_excel,number):
#     '''
#     解析dp决策方案矩阵
#     :param choice_excel: 决策方案矩阵
#     :param number: 初始化选取的最终方案
#     :return:
#     '''
#     for i in choice_excel:
#         if isinstance(i,int):  # 判断是否为整数
#             number.append(i)
#         else:choice(i,number)
#     return number
#
#
# def zeroOneBag(numList,capacity,weightList,valueList):
#     '''
#     动态规划求解01背包
#     :param weightList: 物品重量
#     :param valueList: 物品价值
#     :param numList: 物品数量
#     :return:
#     '''
#     num,newValueList,newWeightList = multi_tran_01(weightList,valueList,numList) # 将多重背包转换成01背包
#     valueExcel= [[0 for j in range(capacity + 1)] for i in range(num + 1)] # 初始化动态规划价值矩阵
#     choicceExcel =[[[] for j in range(capacity + 1)] for i in range(num + 1)]
#     for i in range(1,num+1): #状态转移方程，包括决策序列
#         for j in range(1,capacity+1):
#
#             valueExcel[i][j] = valueExcel[i - 1][j]
#             choicceExcel[i][j].append(choicceExcel[i-1][j])
#             if j >= newWeightList[i-1] and valueExcel[i][j] < (valueExcel[i - 1][j - newWeightList[i - 1]] + newValueList[i - 1]):
#                 valueExcel[i][j] = (valueExcel[i - 1][j - newWeightList[i - 1]] + newValueList[i - 1])
#                 choicceExcel[i][j] = choicceExcel[i - 1][j - newWeightList[i - 1]] + [i]
#     number = []
#     number= choice(choicceExcel[-1][-1],number) #解析决策动态矩阵
#     res = [0 for i in range(len(numList))]
#     for i in number:
#         res[valueList.index(newValueList[i])]+=1
#     return res,valueExcel[-1][-1]
#
#
#
#
# if __name__ == '__main__':
#     package_capacity = 14  # 背包容量66
#     item_weight = [5,3,8,8] # 各类物品重量
#     item_value = [7,6,3,7] # 各类物品价值
#     limit_num = [3,4,6,5] #各类物品限制数量
#     res,maxvalue= zeroOneBag(limit_num, package_capacity, item_weight, item_value) #调用解析程序
#     for i in range(len(item_value)):
#         print(f'总计选取第{i+1}种物品{res[i]}件 ',end='')
#         if i ==len(item_value)-1:
#             print(f'最大物品价值为{maxvalue}')
#
#
#
# # 2. 模拟退火算法的 TSP问题
# import math,random
# def get_city_distance():
#     # 计算两个城市之间的距离
#     for i in range(len(citys)):
#         for j in range(i, len(citys)):
#             d[i][j] = d[j][i] = math.sqrt((citys[i][0] - citys[j][0]) ** 2 + (citys[i][1] - citys[j][1]) ** 2)
# def create_new(a):
#     # 使用随机交换路径中两个城市的位置来产生一条新路径
#     i = int(random.randint(0, len(a) - 2))
#     j = int(random.randint(0, len(a) - 2))
#     a[i], a[j] = a[j], a[i]
#     a[-1]=a[0]
#     return a
#
# def get_route_distance(a):
#     # 获取路径的长度
#     dist = 0
#     for i in range(len(a) - 1):
#         dist += d[a[i]][a[i + 1]]
#     return dist
#
# def saa():
#     get_city_distance()
#     cnt = 0 # 初始化各基础值
#     ans = list(range(0, len(citys)))+[0]
#     print(ans)
#     t = T
#     result = 0
#     while t >= T_end: #退火降温判别是否超温
#         for i in range(0, L):
#             ans_new = create_new(ans) #尝试新的路径
#             d1, d2 = get_route_distance(ans), get_route_distance(ans_new)
#             de = d2 - d1
#             result = d1
#             if de < 0:
#                 ans = ans_new
#                 result = d2
#             else:
#                 if (math.e ** (-de / T) > random.random()):
#                     ans = ans_new
#                     result = d2
#         t = t * delta
#         cnt += 1 # 降温次数
#     return ans,result
#
#
# if __name__ == '__main__':
#     T = 50000 # 初始温度
#     T_end = 1e-8 # 最低温度
#     L = 100 # 在每个温度下的迭代次数
#     delta = 0.98 # 退火系数
#     citys = [[0, 20], [15, 40], [20, 0], [17, 6], [22, 18]] # 5个城市的坐标
#     d = [[0 for i in range(31)] for j in range(31)] # 存储两个城市之间的距离
#
#     ans,result = saa()
#     print(f"路径如下：{ans}")
#     print(f"路径长度：{result}")
#

# 3.蚁群算法的TSP
from math import *
import matplotlib as plt
import random
import sys
import numpy as np
import copy
class Ant(object):
    # 初始化
    def __init__(self,ID,distance_graph,pheromone_graph,ALPHA,BETA,city_index,aim_city,capacity_graph,q_graph):
        self.city_num=len(distance_graph)
        self.distance_graph=distance_graph #距离矩阵
        self.pheromone_graph=pheromone_graph #信息素矩阵
        self.ID = ID  # ID
        (self.ALPHA, self.BETA) = (ALPHA , BETA)#蚂蚁在选择路径时  信息素与距离反比的比重
        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.aim_city = aim_city
        self.initial_city = city_index
        self.current_city = city_index
        self.path.append(city_index)  # 这只蚂蚁经过的路径
        self.move_count = 1
        self.capacity_graph = capacity_graph

    # 选择下一个节点
    def _choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(self.city_num)]  #选择节点的可能性
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(self.city_num):
            if i != self.current_city:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(self.pheromone_graph[self.current_city][i], self.ALPHA) * pow(
                        (1.0 / (self.distance_graph[self.current_city][i]+0.00001)), self.BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,current=self.current_city,target=i))
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.city_num):
                # 轮次相减
                temp_prob -= select_citys_prob[i]
                if temp_prob < 0.0:
                    next_city = i
                    break

        # 未从概率产生，顺序选择一个未访问城市 如果temp_prob恰好选择了total_prob那么就在所有未去的城市中选择一个去的城市
        if self.initial_city == 0:
            if next_city in [0,3,1,7]:
                for i in range(self.city_num):
                    if 0<self.distance_graph[self.current_city][i]<9999:
                        if next_city not in [0, 3, 2, 12]:
                            next_city = i
                            break
        if self.initial_city ==3:
            if next_city in [0,3,2,12,-1]:
                for i in range(self.city_num):
                    if 0<self.distance_graph[self.current_city][i]<9999:
                        if next_city not in [0,3,2,12,-1]:
                            next_city = i
                            break
        # 返回下一个城市序号
        return next_city

    def bpr(self,t,q,c):
        return t*(1+0.15*((q/c)**4))
    # 移动操作
    def _move(self, next_city,q_graph):
        self.path.append(next_city)
        q_graph[self.current_city][next_city] += 1
        self.distance_graph[self.current_city][next_city] = self.bpr(self.distance_graph[self.current_city][next_city],
                                                                q_graph[self.current_city][next_city],self.capacity_graph[self.current_city][next_city])
        self.current_city = next_city
        self.move_count += 1

    def _cal_lenth(self,path):
        temp_distance = 0.0
        for i in range(1, len(path)):
            start, end = path[i], path[i - 1]
            temp_distance += self.distance_graph[start][end]
        return temp_distance

    def _need_reverse(self,start,end):
        tmpPath=self.path[start-1:end+2].copy()
        tmpPath[1:-1]=tmpPath[-2:0:-1]
        return self._cal_lenth(tmpPath) < self._cal_lenth(self.path[start-1:end+2])

    # 搜索路径
    def search_path(self,q_graph):
        # 搜素路径，遍历完所有城市为止
        while self.move_count < self.city_num:
            # 移动到下一个城市
            next_city = self._choice_next_city()
            self._move(next_city,q_graph)
            count = 0
            for i in range(len(self.distance_graph[self.current_city])):
                if self.distance_graph[self.current_city][i] == float('inf'):
                    count+=1
            if self.current_city== self.aim_city:#最后一个城市选择终点城市
                break
            elif count==len(self.distance_graph)-1:

                self.path = [self.initial_city]  # 当前蚂蚁的路径
                self.total_distance = 0.0  # 当前路径的总距离
                self.move_count = 0  # 移动次数
                self.current_city = self.initial_city
                self.move_count = 1
        # 计算路径总长度
        self.total_distance=self._cal_lenth(self.path)





class tsp(object):

    def __init__(self,t,capacity_graph,q_graph,ini,aim,num):#data_set是所有点的经纬度坐标，label_list是这个分组的编号序列
        self.cities = len(t)  # 商店的地址（经纬度信息）
        self.maxIter = 1 #蚁群算法的最大迭代次数
        self.rootNum = len(t)#本分组的商店的数目
        (self.city_num, self.ant_num) = (self.rootNum, num) #每个人是一只蚂蚁
        (self.ALPHA, self.BETA, self.RHO, self.Q) = (1.0, 9.0, 0.5, 100.0)#蚁群算法参数
        self.distance_graph=t#初始化距离
        self.pheromone_graph=[[1.0 for i in range(self.city_num)] for j in range(self.city_num)]
        # self.get_Dis_Pherom()
        self.capacity_graph = capacity_graph
        self.q_graph = q_graph
        self.ini = ini-1
        self.aim = aim-1
        self.new()

    def new(self,evt=None):
        # 初始化信息素
        self.ants = [Ant(ID,self.distance_graph,self.pheromone_graph,self.ALPHA, self.BETA,self.ini,self.aim,self.capacity_graph,self.q_graph) for ID in range(self.ant_num)]
        self.best_ant = self.ants[-1]  # 初始最优解
        self.best_ant.total_distance = (1 << 31)  # 初始最大距离
        self.iter = 0  # 初始化迭代次数


    def search_path(self,evt=None):
        while self.iter<self.maxIter:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path(q_graph)

                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.update_pheromone_gragh()
            #print("迭代次数：", self.iter, u"最佳路径总距离：", int(self.best_ant.total_distance))
            #self.draw()
            self.iter += 1

        self.q_graph[-1][-1]=0
        return self.q_graph


    def update_pheromone_gragh(self):
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(self.city_num)] for raw in range(self.city_num)]
        for ant in self.ants:
            for i in range(1, len(ant.path)):
                # print(ant.path[i - 1], ant.path[i])
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += self.Q / ant.total_distance

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(self.city_num):
            for j in range(self.city_num):
                self.pheromone_graph[i][j] = self.pheromone_graph[i][j] * self.RHO + temp_pheromone[i][j]

def get_route(t,C,q_graph,od13,od42):

    q_graph1 = tsp(t,C,q_graph,1,3,od13).search_path()
    q_graph2 = tsp(t, C, q_graph,4,2,od42).search_path()

    return q_graph2





if __name__ == '__main__':

    od13 = 500
    od42 = 400
    t=np.array([[float('inf') for i in range(13)] for j in range(13)]) #初始化各节点间初始时间矩阵
    for i in range(13):
        t[i][i]=0 #将同一节点的通行时间定义为0
    #输入各个节点1-12的出行时间
    t[0, 11]=20
    t[11, 7]=30
    t[0, 4]=15
    t[11, 5]=20
    t[3, 4]=40
    t[4, 5]=30
    t[5, 6]=18
    t[6, 7]=20
    t[3, 8]=30
    t[4, 8]=20
    t[5, 9]=30
    t[6, 10]=40
    t[7, 1]=10
    t[8, 9]=20
    t[9, 10]=30
    t[10, 1]=10
    t[8, 12]=20
    t[10, 2]=30
    t[12, 2]=40

    # 初始化各节点的通行能力
    C=np.zeros((13, 13))
    for i in range(13):
        C[i, i]=float('inf') #同一节点上的通行能力设为无穷大
    C[0, 11]=200
    C[11, 7]=300
    C[0, 4]=150
    C[11, 5]=200
    C[3, 4]=400
    C[4, 5]=300
    C[5, 6]=180
    C[6, 7]=200
    C[3, 8]=300
    C[4, 8]=200
    C[5, 9]=300
    C[6, 10]=400
    C[7, 1]=100
    C[8, 9]=200
    C[9, 10]=300
    C[10, 1]=100
    C[8, 12]=200
    C[10, 2]=300
    C[12, 2]=400
    q_graph = [[0 for i in range(len(C))] for j in range(len(C))]
    q_graph1 = tsp(t,C,q_graph,1,3,od13).search_path()
    q_graph2 = tsp(t, C, q_graph,4,2,od42).search_path()
    for i in range(len(q_graph)):
        if i ==0:
            print('分配后交通量矩阵如下：')
        print(q_graph2[i])
#

# 5.生产线性规划
#
# import pulp
#
# def fun_bulit(V_NUM,c,cost_mat,constract,fix=0,item_num=['Ⅰ','Ⅱ','Ⅲ']):
#     '''
#
#     :param V_NUM: 变量数量，这里就是产品数量
#     :param c: 价值矩阵，即单位产品的收益
#     :param cost_mat: 产品耗时矩阵
#     :param constract: 约束条件值，即每个设备工时
#     :param fix: 修正值，对应在目标函数下的与真实值的偏差
#     :param item_num: 产品的编号
#     :return:
#     '''
#     variables = [pulp.LpVariable('X%d' % i, lowBound=0, cat=pulp.LpInteger) for i in range(0, V_NUM)]  # 初始化变量并完成非负约束
#     objective = sum([c[i] * variables[i] for i in range(0, V_NUM)])
#     constraints = []  # 增加三个约束条件
#     for j in range(3):
#         constraints.append(sum([cost_mat[j][i] * variables[i] for i in range(0, V_NUM)]) <= constract[j])
#
#     prob = pulp.LpProblem('LP1', pulp.LpMaximize)  # 初始化计算器，LP1为计算模式（整数线性规划），LpMaximize确定规划目标为最大化
#     prob += objective  # 给计算器配置计算目标
#     for cons in constraints:  # 给计算器配置约束条件
#         prob += cons
#     status = prob.solve()  # 调用计算器
#
#     if status != 1:
#         # 如果计算无解则返回无解
#         return None
#     else:
#         res=[v.varValue.real for v in prob.variables()]  # 否则返回计算器中的最优变量
#
#     for i in range(V_NUM):
#         print(f'生产{item_num[i]}产品{res[i]}件 ',end='')
#     print(f"总利润为{sum([c[i] * res[i] for i in range(0, V_NUM)])-fix}千元")
#
# if __name__ == '__main__':
#     print('问题1:')
#     fun_bulit(V_NUM=3,c=[3, 2, 2.9],cost_mat=[[8, 2, 10],[10, 5, 8],[2, 13, 10]],constract = [300,400,420])
#     print('问题2:')
#     fun_bulit(V_NUM=3,c=[3, 2, 2.9],cost_mat=[[8, 2, 10],[10, 5, 8],[2, 13, 10]],constract = [300,460,420],fix=18)
#     print('问题3:')
#     fun_bulit(V_NUM=4,c=[3, 2, 2.9,2.1],cost_mat=[[8, 2, 10,12],[10, 5, 8,5],[2, 13, 10,10]],constract = [300,400,420],item_num=['Ⅰ','Ⅱ','Ⅲ','Ⅳ'])
#     fun_bulit(V_NUM=4,c=[3, 2, 2.9,1.87],cost_mat=[[8, 2, 10,4],[10, 5, 8,4],[2, 13, 10,12]],constract = [300,400,420],item_num=['Ⅰ','Ⅱ','Ⅲ','Ⅴ'])
#     print('问题4:')
#     fun_bulit(V_NUM=3,c=[4.5, 2, 2.9],cost_mat=[[9, 2, 10],[12, 5, 8],[4, 13, 10]],constract = [300,400,420])\



# import numpy as np
# import math
#
# def MSA_Logit(t0,p,q,sita,va,C,path0,path1,path2,path3,path4,path5,le):
#     #MSA算法求解Logit模型
#     ta=np.zeros((14,14)) #定义路段阻抗
#     for i in range(0,14):
#         for j in range(0,14):
#             if C[i,j]==0:
#                 ta[i,j]=float('inf')
#             else:
#                 ta[i,j]=t0[i,j]*(1+p*pow(va[i,j]/C[i,j],q)) #
#                 #采用BPR函数计算路段阻抗
#     crs=np.zeros(6)
#     for a in range(0,le[0]-1):
#         crs[0]=crs[0]+ta[path0[a],path0[a+1]]
#     for b in range(0,le[1]-1):
#         crs[1]=crs[1]+ta[path1[b],path1[b+1]]
#     for c in range(0,le[2]-1):
#         crs[2]=crs[2]+ta[path2[c],path2[c+1]]
#     for d in range(0,le[3]-1):
#         crs[3]=crs[3]+ta[path3[d],path3[d+1]]
#     for e in range(0,le[4]-1):
#         crs[4]=crs[4]+ta[path4[e],path4[e+1]]
#     for f in range(0,le[5]-1):
#         crs[5]=crs[5]+ta[path5[f],path5[f+1]]
#     #根据路段阻抗计算各条路径阻抗
#     n=0
#     for g in range(0,6):
#         n=n+math.exp(-sita*crs[g])
#     pk=np.zeros(6) #定义路径选择概率
#     fk=np.zeros(6) #定义各条路径上的流量
#     for h in range(0,6):
#         pk[h]=(math.exp(-sita*crs[h]))/n #采用Logit模型路径选择概率公式计算选择各条路径的概率
#         fk[h]=pk[h]*Q #计算Logit形式的路径流量
#     va00=np.zeros((14,14))
#     va01=np.zeros((14,14))
#     va02=np.zeros((14,14))
#     va03=np.zeros((14,14))
#     va04=np.zeros((14,14))
#     va05=np.zeros((14,14))
#     #定义各路段流量
#     for u in range(0,le[0]-1):
#         va00[path0[u],path0[u+1]]=fk[0]
#     for v in range(0,le[1]-1):
#         va01[path1[v],path1[v+1]]=fk[1]
#     for w in range(0,le[2]-1):
#         va02[path2[w],path2[w+1]]=fk[2]
#     for x in range(0,le[3]-1):
#         va03[path3[x],path3[x+1]]=fk[3]
#     for y in range(0,le[4]-1):
#         va04[path4[y],path4[y+1]]=fk[4]
#     for z in range(0,le[5]-1):
#         va05[path5[z],path5[z+1]]=fk[5]
#     #将路径流量表示到各路段上
#     va=np.zeros((14,14))
#     va=va00+va01+va02+va03+va04+va05 #计算路段流量
#     return va
#
#
#
# t=np.array([[float('inf') for i in range(14)] for j in range(14)]) #初始化各节点间初始时间矩阵
# for i in range(1,14):
#     t[i][i]=0 #将同一节点的通行时间定义为0
# #输入各个节点1-12的出行时间
# t[1, 12]=20
# t[12, 8]=30
# t[1, 5]=15
# t[12, 6]=20
# t[4, 5]=40
# t[5, 6]=30
# t[6, 7]=18
# t[7, 8]=20
# t[4, 9]=30
# t[5, 9]=20
# t[6, 10]=30
# t[7, 11]=40
# t[8, 2]=10
# t[9, 10]=20
# t[10, 11]=30
# t[11, 2]=10
# t[9, 13]=20
# t[11, 3]=30
# t[13, 3]=40
#
#
# # 初始化各节点的通行能力
# C=np.zeros((14, 14))
# for i in range(1,14):
#     C[i, i]=float('inf') #同一节点上的通行能力设为无穷大
# C[1, 12]=150
# C[12, 8]=200
# C[1, 5]=150
# C[12, 6]=100
# C[4, 5]=200
# C[5, 6]=200
# C[6, 7]=150
# C[7, 8]=200
# C[4, 9]=200
# C[5, 9]=100
# C[6, 10]=200
# C[7, 11]=300
# C[8, 2]=200
# C[9, 10]=300
# C[10, 11]=200
# C[11, 2]=300
# C[9, 13]=200
# C[11, 3]=300
# C[13, 3]=200
# C[0, 0]=0
#
# Q=150# 初始化1-3节点的od量
# path0=[1,5,6,7,11,3]
# le0=len(path0)
# path1=[1,5,6,10,11,3]
# le1=len(path1)
# path2=[1,5,9,10,11,3]
# le2=len(path2)
# path3=[1,5,9,13,3]
# le3=len(path3)
# path4=[1,12,6,7,11,3]
# le4=len(path4)
# path5=[1,12,6,10,11,3]
# le5=len(path5)
# le=[le0,le1,le2,le3,le4,le5]
# #确定OD对1-3之间的有效路径及各条有效路径所包含的节点数量
# p=0.15
# q=4
# #BPR函数中两个校正参数p取0.15，q取4
# sita=3 #Logit模型路径选择概率公式中常参数θ设为3
# va0=np.zeros((14,14)) #路段初始交通量设为0
# vai=[]
# van=MSA_Logit(t, p, q, sita, va0, C, path0, path1, path2, path3, path4, path5, le)
# '''根据路径阻抗对给定的OD交通量执行一次Logit模型加载，得到各路径流量和各路段交通量，
# 其中各路段初始自由阻抗与路段出行时间成正比'''
# vai.append(van)
# vi0=0
# for f in range(0,14):
#     for g in range(0,14):
#         vi0=vi0+pow(van[f,g],2)
# RSME=np.sqrt(vi0/19) #初始化均方根误差
# n=0
# while RSME>=0.001: #用均方根误差判断算法的收敛性
#     n=n+1
#     ya=MSA_Logit(t, p, q, sita, van, C, path0, path1, path2, path3, path4, path5, le)
#     #执行Logit模型加载,更新各路段阻抗、路径阻抗和各路段附加交通流量
#     van=vai[n-1]+(1/n)*(ya-vai[n-1])
#     '''将上次循环中各路段的交通流量与本次循环中所获得的附加交通流量进行加权平均，得到本次循环
#     各路段的交通流量，采用的标准步长公式为a(n)=1/n'''
#     vai.append(van)
#     va=np.array(vai)
#     vi=0
#     for h in range(0,14):
#         for i in range(0,14):
#             vi=vi+pow(va[n,h,i]-va[n-1,h,i],2)
#     RSME=np.sqrt(vi/19) #计算均方根误差
# van=va[n] #路段最终交通量为最后一次迭代所获得的交通量
# tan=np.zeros((14,14))
# for j in range(0,14):
#     for k in range(0,14):
#         if C[j, k]==0:
#             tan[j,k]=float('inf')
#         else:
#             tan[j,k]= t[j, k] * (1 + p * pow(van[j, k] / C[j, k], q)) #计算最终的路段阻抗
# print("交通流量分配结果为：\n",van) #输出各路段交通流量
# print("路段实际阻抗为：\n",tan) #输出各路段阻抗

