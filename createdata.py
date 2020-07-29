import random

trainfile = open("trainfile.txt","w",encoding="utf-8")
trainY = open("trainY.txt","w",encoding="utf-8")
count = [0,0,0,0]
for i in range(50000):
    x1 = random.random() * 10
    x2 = random.random() * 10
    x3 = random.random() * 10
    x4 = random.random() * 10
    trainfile.write("{} {} {} {}\n".format(x1, x2, x3, x4))
    f1 = x1 + x2 + x3 + x4
    f2 = x1 - x2 + x3 - x4
    if f1 < 12.5 or f1 > 25 :
        trainY.write("{} {} {} {}\n".format(1, 0, 0, 0))
        count[0] = count[0] + 1
    elif f2 < -2.5 :
        trainY.write("{} {} {} {}\n".format(0, 1, 0, 0))
        count[1] = count[1] + 1
    elif f2 > 5 :
        trainY.write("{} {} {} {}\n".format(0, 0, 1, 0))
        count[2] = count[3] + 1
    else:
        trainY.write("{} {} {} {}\n".format(0, 0, 0, 1))
        count[3] = count[3] + 1
print(count)
