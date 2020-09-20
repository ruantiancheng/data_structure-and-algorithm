# # 第一题
#
# # 时间复杂度为O（1）散列表
# class HashTable:
#     def __init__(self):
#         self.size = 9
#         self.slots = [None] * self.size  # hold the key items
#         self.data = [0] * self.size  # hold the data values
#
#     def hashfunction(self, key, size):
#         return key % size
#
#     def rehash(self, oldhash, size):
#         return (oldhash + 1) % size
#
#     def put(self, key, data):
#         hashvalue = self.hashfunction(key, len(self.slots))
#         if self.slots[hashvalue] == None:  # 如果slot内是empty，就存进去
#             self.slots[hashvalue] = key
#             self.data[hashvalue] = data
#         else:  # slot内已有key
#             if self.slots[hashvalue] == key:  # 如果已有值等于key,更新data
#                 self.data[hashvalue]  += 1  # replace
#             else:  # 如果slot不等于key,找下一个为None的地方
#                 nextslot = self.rehash(hashvalue, len(self.slots))
#                 while self.slots[nextslot] != None and self.slots[nextslot] != key:
#                     nextslot = self.rehash(nextslot, len(self.slots))
#                     print('while nextslot:', nextslot)
#                 if self.slots[nextslot] == None:
#                     self.slots[nextslot] = key
#                     self.data[nextslot] = data
#                     print('slots None')
#                 else:
#                     self.data[nextslot] = data
#                     print('slots not None')
#
#     def get(self, key):
#         startslot = self.hashfunction(key, len(self.slots))
#         data = None
#         stop = False
#         found = False
#         position = startslot
#         while self.slots[position] != None and not found and not stop:
#             if self.slots[position] == key:
#                 found = True
#                 data = self.data[position]
#             else:
#                 position = self.rehash(position, len(self.slots))
#                 if position == startslot:
#                     stop = True
#         return data
#
#     def __getitem__(self, key):
#         return self.get(key)
#
#     def __setitem__(self, key, data):
#         print('key:', key)
#         print('data:', data)
#         self.put(key, data)
# def three_dif_item(list):
#     pop_list = []
#     pop_list.append(list.pop())
#     for i in range(len(list)):
#         if len(list) <= 2 or len(pop_list) ==3:
#             return list
#         work_element = list[i]
#         if work_element not in pop_list:
#             pop_list.append(list[i])
#             list.remove(list[i])
#
#
# def first_question():
#     aim_list = [1, 5, 7, 5, 9, 5, 4, 2, 5,10]
#     h = HashTable()
#     for i in aim_list:
#         h[i]=1
#     print(h.slots[h.data.index(max(h.data))])
#
#     # 时间复杂度为O（n）list：
#     print(aim_list)
#     for i in range(len(aim_list)//3):
#         list = three_dif_item(aim_list)
#     print(int(np.mean(list)))
#
# first_question()


# # 第二题
# def second_question():
#     aim_list = [3,1,6,4,5,7,9,8,10,14,12]
#     activate = [False]*len(aim_list)
#
#     st = Stack()
#     st2= Stack()
#     for i in aim_list:
#         if st.isEmpty():
#             st.push(i)
#         if i > st.peek():
#             st.push(i)
#     for i in reversed(aim_list):
#         if st2.isEmpty():
#             st2.push(i)
#         if i < st2.peek():
#             st2.push(i)
#     print(st.size(),st2.size())
#     a = []
#     b= []
#     for i in range(st.size()):
#         a.append(st.pop())
#     for i in range(st2.size()):
#         b.append(st2.pop())
#     print(a,b)
#     print([i for i in a if i in b])
#
# second_question()


# # 第三题
# class LNode:
#     def __init__(self, item):
#         self.data = item
#         self.next = None
#
#
# def FindMiddleNode(head):
#     if head is None or head.next is None:
#         return head
#     fast = head
#     slow = head
#     while fast is not None and fast.next is not None:
#         slowpre = slow
#         slow = slow.next
#         fast = fast.next.next
#     slowpre.next = None
#     return slow
#
#
# # 对不带头单链表进行翻转
# def Reverse(head):
#     if head is None or head.next is None:
#         return head
#     pre = head
#     cur = head.next
#     next = cur.next
#     pre.next = None
#     while cur is not None:
#         next = cur.next
#         cur.next = pre
#         pre = cur
#         cur = next
#     return pre
#
#
# # 对联表进行排序
# def Resort(head):
#     if head is None or head.next is None:
#         return
#     cur1 = head.next
#     mid = FindMiddleNode(head)
#     cur2 = Reverse(mid)
#     tmp = None
#     # 合并两个链表
#     while cur1.next is not None:
#         tmp1 = cur1.next
#         tmp2 = cur2.next
#         cur1.next = cur2
#         cur1 = tmp1
#         cur2.next = cur1
#         cur2 = tmp2
#
#     cur1.next = cur2
#
#
# if __name__ == "__main__":
#     i = 1
#     head = LNode(None)
#     head.next = None
#     tmp = None
#     cur = head
#     while i < 5:
#         tmp = LNode(i)
#         tmp.next = None
#         cur.next = tmp
#         cur = tmp
#         i += 1
#     print("排序前：")
#     cur = head.next
#     while cur != None:
#         print(cur.data)
#         cur = cur.next
#     Resort(head)
#     print("排序后：")
#     cur = head.next
#     while cur is not None:
#         print(cur.data)
#         cur = cur.next

# # 第四题
# def fourth_question():
#     # 初始化问题
#     aim_list = [1,5,8,4,9]
#     out_list = []
#     st = Stack()
#     st2 = Stack()
#     for i in aim_list:
#         st.push(i)
#     st2.push(st.pop())
#     while not st.isEmpty():
#         temp = st.pop()
#         if st2.peek() < temp:
#             st2.push(temp)
#         else:
#
#
#
#             while not st2.isEmpty():
#                 if st2.peek()>temp:
#                  st.push(st2.pop())
#                 else:
#                     st.push(st2.pop())
#             while st2.isEmpty():
#                 st2.push(temp)
#
#     for i in range(st2.size()):
#         out_list.append(st2.pop())
#     print(out_list)
#
# fourth_question()


# # 第五题
# def mins(a, b, c):
#     mins = a if a < b else b
#     mins = mins if mins < c else c
#     return mins
#
#
# def maxs(a, b, c):
#     maxs = b if a < b else a
#     maxs = c if maxs < c else maxs
#     return maxs
#
#
# def fifth(a, b, c):
#     aLen = len(a)
#     bLen = len(b)
#     cLen = len(c)
#     curDist = 0
#     minsd = 0
#     minDist = 2 ** 32
#     i = 0  # 数组a的下标
#     j = 0  # 数组b的下标
#     k = 0  # 数组c的下标
#     while True:
#         curDist = maxs(abs(a[i] - b[j]), abs(a[i] - c[k]), abs(b[j] - c[k]))
#         if curDist < minDist:
#             minDist = curDist
#             min_list = [a[i], b[j], c[k]]
#             # 找出当前遍历到三个数组中最小值
#         minsd = mins(a[i], b[j], c[k])
#         if minsd == a[i]:
#             i += 1
#             if i >= aLen:
#                 break
#         elif minsd == b[j]:
#             j += 1
#             if j >= bLen:
#                 break
#         else:
#             k += 1
#             if k >= cLen:
#                 break
#     return minDist,min_list
#
#
# if __name__ == "__main__":
#     a = [3, 4, 5, 15, 18]
#     b = [10, 12, 14, 16, 20]
#     c = [17, 21, 23, 24, 37, 30]
#     mindis,min_list = fifth(a,b,c)
#     print("最小距离为：",mindis,'最小三元组为',min_list)


# # 第六题
# def sixth(num_list):#动态规划求最大连续子序列
#     length=len(num_list)
#     max_value=-10000000
#     tmp=0
#     sub_list = []
#     for i in range(length):
#         if tmp+num_list[i]>num_list[i]:
#             tmp = tmp+num_list[i]
#             sub_list.append(num_list[i-1])
#         else:
#             tmp = num_list[i]
#             sub_list = []
#         if max_value<tmp:
#             max_value = tmp
#
#     return max_value,sub_list
# count_list = [1,-2,3,10,-4,7,2,-5]
# max_value , sub_list = sixth(count_list)
# print(max_value , sub_list)


# # 第七题
# def maxSubArray(nums):
#     if len(nums) < 2:
#         return nums[0]
#     tem = nums[0]
#     max_num = nums[0]
#     for i in range(1, len(nums)):
#         if tem < 0:
#             tem = nums[i]
#         else:
#             tem += nums[i]
#         if tem > max_num:
#             max_num = tem
#
#     return max_num
#
#
# # 将一个矩阵分成数组的形式
# def seventh(matrix):
#     if not matrix:
#         return
#     low = len(matrix)
#     max_num = 0
#     for i in range(low):
#         res1 = list(matrix[i])
#         for j in range(i + 1, low):
#             res1 = list(map(lambda x: x[0] + x[1], zip(res1, matrix[j])))
#             max_num = max(maxSubArray(res1), max_num)
#     return max_num
#
#
#
# ary = [[0, -2, -7, 0], [9, 2, -6, 2], [-4, 1, -4, 1], [-1, 8, 0, -2]]
# print(seventh(ary))

# # 第八题
#
# def eighth(string):
#     cnt = [0] * len(string)
#     flag = 0
#     maxr = 0
#
#     for i in range(1,len(string),2):
#         activate = True
#         # if maxr > i:
#         #     cnt[i] = min(cnt[2 * flag - i], maxr - i)
#         # else:
#         #     cnt[i] = 1
#
#         while activate:
#             if string[i + cnt[i]] == string[i - cnt[i]]:
#
#                 cnt[i] += 1
#                 if i + cnt[i]>=len(string): activate=False
#             else: activate = False
#
#
#     return int(max(cnt)/2+1), string[cnt.index(max(cnt))]
#
#
# string = '#1#2#3#4#3#1#'
# maxr,flag= eighth(string)
# print(maxr,flag)
