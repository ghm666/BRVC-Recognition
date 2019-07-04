# 生成caffe文件标签
"""
train----- [1] [2]
"""
import os
def genetareLabel(dir):
    listText = open('val11111.txt', 'w')
    files = os.listdir(dir)
    for subfiles in files:
        Path = os.path.join(dir, subfiles)
        newfiles = os.listdir(Path)
        for filesub in newfiles:
            name = subfiles + "/" + filesub + ' ' + str(int(subfiles)) + '\n'
            listText.write(name)
    listText.close()
    print("Done...........")

if __name__ == '__main__':
    dir = "F:/doctor_labeled_3class/data/validation"
    genetareLabel(dir)




