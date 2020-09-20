# # 排序第一题
#
# def buble_sort(list):
#     for i in range(len(list)-1,0,-1):
#         for j in range(i):
#             if list[j]%2 == list[j+1]%2 ==1:
#                 pass
#             elif list[j]%2 < list[j+1]%2:
#                 list[j],list[j+1]=list[j+1],list[j]
#     print(list)
#
# list= [3,2,6,1,4,9,8,5,11,7]
#
# buble_sort(list)


# # 排序第二题
#
# def bubblesec(mat):
#     mat= [[22,44,87,50,18,42],
#           [21,96,25,47,42,21],
#           [17,42,46,54,78,29],
#           [52,43,89,42,27,39]]
#
#     for epoc in range(len(mat)-1):
#         for i in range(len(mat)-1):
#             for j in range(len(mat[i])-1):
#                 if mat[i][j] > mat[i + 1][j]:
#                     mat[i][j], mat[i + 1][j] = mat[i + 1][j], mat[i][j]
#     for i in range(len(mat)):
#         for epoc2 in range(len(mat[i]) - 1):
#             for j in range(len(mat[i])-1):
#                 if mat[i][j]>mat[i][j+1]:
#                     mat[i][j],mat[i][j+1]=mat[i][j+1],mat[i][j]
#
#
# mat= [[22,44,87,50,18,42],
#       [21,96,25,47,42,21],
#       [17,42,46,54,78,29],
#       [52,43,89,42,27,39]]
# mat = bubblesec(mat)
# print(mat)



# # 排序第三题
#
# class Node():
#     def __init__(self, item=None):
#         self.item = item
#         self.next = None
#
# class LinkList():
#     def __init__(self):
#         self.head = None
#
#     def create(self, item):
#         self.head = Node(item[0])
#         p = self.head
#         for i in item[1:]:
#             p.next = Node(i)
#             p = p.next
#
#     def print(self):
#         p = self.head
#         while p != None:
#             print(p.item, end=' ')
#             p = p.next
#         print()
#
#     def getItem(self, index):
#         p = self.head
#         count = 0
#         while count != index:
#             p = p.next
#             count += 1
#         return p.item
#
#
#     def setItem(self, index, item):
#         p = self.head
#         count = -1
#         while count < index - 1:
#             p = p.next
#             count += 1
#         p.item = item
#
#
#     def swapItem(self, i, j):
#         t = self.getItem(j)
#         self.setItem(j, self.getItem(i))
#         self.setItem(i, t)
#
#     def quicksortofloop(self, left, right):
#         if left < right:
#             i = left
#             j = i + 1
#             start = self.getItem(i)
#
#             while (j <= right):
#
#                 while (j <= right and self.getItem(j) >= start):
#                     j += 1
#
#                 if (j <= right):
#                     i += 1
#                     self.swapItem(i, j)
#                     self.print()
#                     j += 1
#             self.swapItem(left, i)
#             self.quicksortofloop(left, i - 1)
#             self.quicksortofloop(i + 1, right)
#
# if __name__ == "__main__":
#     L = LinkList()
#     L.create([4, 2, 5, 3, 7, 9, 0, 1])
#     L.quicksortofloop(0, 7)
#     L.print()

# # 排序第四题
# def partition(li, low, high):
#     #首先设置俩个布尔变量，通过这个来控制左右移动
#     high_flag = True
#     low_flag = False
#     #将开始位置的值定为基数
#     pivot = li[low]
#     while low < high and low < len(li) and high < len(li):
#         #当这个值为真时，游标从右开始移动
#         if high_flag:
#             #找出右边比基数小的值，互换位置，否则一直向右移动
#             if li[high] < pivot:
#                 li[low] = li[high]
#                 #改变布尔值，控制方向
#                 high_flag = False
#                 low_flag = True
#             else:
#                 high -= 1
#         if low_flag:
#             if li[low] > pivot:
#                 li[high] = li[low]
#                 high_flag = True
#                 low_flag = False
#             else:
#                 low += 1
#     li[low] = pivot
#     #返回的是索引位置
#     return low
#
#
# def quickSort(li):
#     arr = []
#     low = 0
#     high = len(li) - 1
#     if low < high:
#         #mid是确定位置的索引
#         mid = partition(li, low, high)
#         #确定值左边
#         if low < mid - 1:
#             #将左边区的第一和最后数索引放进去
#             arr.append(low)
#             arr.append(mid - 1)
#         #确定值的右边
#         if mid + 1 < high:
#             arr.append(mid + 1)
#             arr.append(high)
#         #循环
#         while arr:
#             #依次取出一个区域的范围索引
#             r = arr.pop()
#             l = arr.pop()
#             #重复上面的找出该区域的可以确定下来的一个值的索引
#             mid = partition(li, l, r)
#             if l < mid - 1:
#                 arr.append(l)
#                 arr.append(mid - 1)
#             if mid + 1 < r:
#                 arr.append(mid + 1)
#                 arr.append(r)
#     return li
#
# a = quickSort([3,1,6,4,9,5,7,11])
# print(a)

# # 排序第五题
# def solution(num, k):
#     s = str(num)
#     flag = True
#     while k:
#         for i in range(len(s) - 1):
#             # 每次删除第一个比下一个数字大的数
#             if s[i] > s[i + 1]:
#                 s = s.replace(s[i], '', 1)
#                 flag = False
#                 break
#
#         # 如果所有数字递增，则删除最后几个数字直接返回
#         if flag:
#             s = s[:len(s) - k]
#         k -= 1
#     return int(s)
#
# a= 1432219
# print(solution(a,4))

# # 排序第六题 合并石堆dp
# class Solution(object):
#     def getSum(self, i, j):
#         return self.s[j] - self.s[i - 1]
#
#     def mergeStone(self, nums):
#         '''
#         nums : List[int]
#         '''
#         n = len(nums)
#         nums = [0] + nums + nums
#         L = 1
#         R = 2 * n
#         s = [i for i in nums]
#         for i in range(1, len(s)):
#             # 前缀和
#             s[i] = s[i - 1] + s[i]
#         self.s = s
#         # extra 1 for lower limit.
#         self.min_memo = [[None for b in range(n + n + 1)] for a in range(n + n + 1)]
#
#         self.dp_min(L, R)
#         minres = float('inf')
#         for i in range(1, n + 1):
#             minres = min(minres, self.min_memo[i][i + n - 1])
#         print(self.min_memo)
#         print(minres)
#
#     def dp_min(self, i, j):
#         '''
#         i : 区间始
#         j : 区间末
#         s : 前缀和，用于访问
#         '''
#         if j == i:
#             self.min_memo[i][j] = 0
#             return 0
#         elif self.min_memo[i][j] != None:
#             return self.min_memo[i][j]
#         else:
#             this_cost = self.getSum(i, j)
#             tres = float('inf')
#             for c in range(i, j):
#                 tres = min(tres, self.dp_min(i, c) + self.dp_min(c + 1, j) + this_cost)
#             self.min_memo[i][j] = tres
#             return tres
# ab = Solution()
#
# ab.mergeStone([5,2,3,4,1,6])

# # 排序第七题 构造回文串
# class Solution(object):
#     def generate_str(self,str):
#         n = len(str)
#         self.cost = [[0 for i in range(n+1) ]for j in range(n+1)]
#         for i in range(n-1,-1,-1):
#             for j in range(i,n):
#                 if str[i]==str[j]:
#                     self.cost[i][j] = self.cost[i+1][j-1]
#                 else:
#                     self.cost[i][j] = min(1+self.cost[i][j-1],1+self.cost[i+1][j])
#
#
#         print(self.cost[0][n-1])
#
#
# solution = Solution()
# solution.generate_str('12345678')

# # 排序第八题 月影定理
# class Solution(object):
#     def max_product(self,n):
#         self.product = [0 for i in range(n+1)]
#         for i in range(1,n+1):
#             if i == 1:
#                 self.product[i] = 1
#             elif i == 2:
#                 self.product[i] = 2
#             elif i == 3:
#                 self.product[i] = 3
#             else:
#                 self.product[i] = max(self.product[i-2]*2,self.product[i-3]*3)
#
#
#         print(self.product[-1])
#
#
#
# solution =Solution()
# solution.max_product(7)

