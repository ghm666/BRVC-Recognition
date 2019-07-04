# -*- coding:utf-8 -*-

"""
本文档主要实现图片文件名修改前和修改后的对应
结果：dice={key-value}
"""
import os

# filename_index = 0,1,2
filename_index = 2
file_name = "F:\\brvo_crvo_nomal_dataset\\doctor_label_data\\newdata"
sub_file_name = ["brvo","crvo","normal"]
image_file = os.path.join(file_name,sub_file_name[filename_index])
image_name = os.listdir(image_file)
name2number = { }

count = 330
for i in image_name:
    # print(i + ' '+ 'brvo'+str(count))
    name2number[i] = sub_file_name[filename_index]+str(count)
    count += 1


# 保存
f = open(file_name+"\\"+sub_file_name[filename_index]+'_dict.txt', 'w',encoding= "utf-8")
f.write(str(name2number))
f.close()
print("done...........")

# 读取
# f = open('temp.txt', 'r',encoding= "utf-8")
# a = f.read()
# dict_name = eval(a)
# f.close()
# # print(dict_name)


# 判断一张图片是否在字典中
# print(name2number['1401072张XYWA.jpg'])

# a = name2number.get('14122',-1)
# print(a)







