# 对图片进行重命名
import os

fileDir = "F:\\brvo_crvo_nomal_dataset\\doctor_label_data\\newdata"
sub_name = ["brvo1","crvo1","normal1"]
# sub_name[0] //  sub_name[1]  //sub_name[2]
imageDir = os.path.join(fileDir,sub_name[2])
print(imageDir)

imageName = os.listdir(imageDir)
# imageName.sort(key=lambda x:int(x[:-4]))

i = 1
for imagename in imageName:
     print(imagename)
     # newname = imageDir+"/"+str(i)+".json"
     newname = imageDir +"/normal"+ str(i) + ".jpg"
     oriname = imageDir+"/"+imagename
     os.rename(oriname,newname)
     i+=1