# 打乱txt中的标签顺序
from numpy import *

# 打开原txt
fr = open("valnew.txt")
lineArr = []
for line in fr.readlines():
    lineArr.append(line)
# print(lineArr)
m= len(lineArr)
A = list(range(m))
random.shuffle(A)

# 新打乱的列表
newtrainX = []
for j in range(m):
    newtrainX.append(lineArr[A[j]])
fr.close()

# 存入新的txt
newfile = open("val1.txt","w")
for j in range(m):
    newfile.write(newtrainX[j])
newfile.close()



