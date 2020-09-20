# # 综合二 计算24点
# def calculates(number, n, result):
#     if n == 1:
#         if number[0] == result:
#             print(number[0], end='')
#             return True
#
#     else:
#         if calculates(number, n - 1, result - number[n - 1]):
#             print(' + ', number[n - 1], end='')
#             return True
#         if calculates(number, n - 1, result + number[n - 1]):
#             print(' - ', number[n - 1], end="")
#             return True
#         if calculates(number, n - 1, result / number[n - 1]):
#             print(' * ', number[n - 1], end='')
#             return True
#         if calculates(number, n - 1, result * number[n - 1]):
#             print(' / ', number[n - 1], end='')
#             return True
#     return False
#
#
# number = [12, 2, 6, 3]  # 这里填四张牌的数字
# calculates(number, 4, 24)


# # 综合三 骑士周游问题
# import datetime
# from enum import Enum
#
#
# class Size(Enum):
#     X = 8
#
#
# start = datetime.datetime.now()
# chess = [[0 for i in range(Size.X.value)] for j in range(Size.X.value)]
#
#
# def nextXY(x, y, position):
#     global chess
#     if position == 0 and x - 2 >= 0 and y - 1 >= 0 and chess[x - 2][y - 1] == 0:
#         return [1, x - 2, y - 1]
#     elif position == 1 and x - 2 >= 0 and y + 1 <= Size.X.value - 1 and chess[x - 2][y + 1] == 0:
#         return [1, x - 2, y + 1]
#     elif position == 2 and x - 1 >= 0 and y - 2 >= 0 and chess[x - 1][y - 2] == 0:
#         return [1, x - 1, y - 2]
#     elif position == 3 and x - 1 >= 0 and y + 2 <= Size.X.value - 1 and chess[x - 1][y + 2] == 0:
#         return [1, x - 1, y + 2]
#     elif position == 4 and x + 1 <= Size.X.value - 1 and y - 2 >= 0 and chess[x + 1][y - 2] == 0:
#         return [1, x + 1, y - 2]
#     elif position == 5 and x + 1 <= Size.X.value - 1 and y + 2 <= Size.X.value - 1 and chess[x + 1][y + 2] == 0:
#         return [1, x + 1, y + 2]
#     elif position == 6 and x + 2 <= Size.X.value - 1 and y - 1 >= 0 and chess[x + 2][y - 1] == 0:
#         return [1, x + 2, y - 1]
#     elif position == 7 and x + 2 <= Size.X.value - 1 and y + 1 <= Size.X.value - 1 and chess[x + 2][y + 1] == 0:
#         return [1, x + 2, y + 1]
#     else:
#         return [0, x, y]
#
#
# def TravelChessBoard(x, y, tag):
#     global chess
#     chess[x][y] = tag
#     if tag == Size.X.value ** 2:
#         for i in chess:
#             print(i)
#         return "OK"
#     f = 0
#     for i in range(8):
#         flag = nextXY(x, y, i)
#         if flag[0]:
#             statues = TravelChessBoard(flag[1], flag[2], tag + 1)
#             if statues == "OK":
#                 return "OK"
#             f += 1
#         else:
#             f += 1
#     if f == 8:
#         chess[x][y] = 0
#
#
# print(TravelChessBoard(2, 0, 1))
# print(datetime.datetime.now() - start)
