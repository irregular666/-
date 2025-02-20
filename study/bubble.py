
x = input("请输入一些数（中间用空格隔开）：")

x = list(map(int , x.split()))

for i in range(0 , len(x) - 1):

    for j in range(0 , len(x) - i - 1):

        if x[j] >= x[j + 1]:

           x[j] , x[j + 1] = x[j + 1] , x[j]




print(x)
