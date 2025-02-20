"""商店首充活动及充值优惠
money=input("您所充值的金额为：")
money=int(money)
first_name=input("请输入你的姓：")

if(first_name=="金"):
    money=money+money
else:
    money=money





if money<=1000:
    print("您现在卡中余额为:",money)
elif money<=2000:
    money=money+money*0.15
    print("您现在卡中余额为:", money)

elif money<=10000:
    money=money+money*0.2+500
    print("您现在卡中余额为:", money)

else :
    money=money+10000
    print("您现在卡中余额为:", money)"""


"""电脑自己猜数字mport random
right=random.randint(1,100000)
guess=0
count=0
low=1
high=100000
while right!=guess:
    guess = random.randint(low, high)
    count+=1
    if guess>right:
        high=guess
    elif guess<right:
        low=guess
    else:
        print(f"正确的数字是{right},我猜了{count}次")"""














"""偶数求和
total=0
for i in range(2,10000,2):
    total+=i
print(total)"""

















"""九九乘法表
for i in range(1,10):
    if i==1:
        continue
    else:
        for j in range(1,i+1):
            #若无此行，九九乘法表会出现重复


            print(f"{i}*{j}={i*j}",end='\t')
        print()      #换行，等价于print("\n")"""



"""import random
right=random.randint(1,100000000)
low_number=0
high_number=100000000



guess_time=0



while middle!=right:
    if middle>right:
        high_number=middle
        middle=(low_number + high_number) / 2
        guess_time+=1

    elif middle<right:
        low_number=middle
        middle = (low_number + high_number) / 2
        guess_time+=1

print(f"我猜了{guess_time}次，正确的数字是{right}")"""



"""import random

def calc_middle(low_number, high_number):
    return int((low_number + high_number) / 2)

def guess(low_number, high_number):
    right = random.randint(low_number, high_number)  # 在函数内部生成随机数
    guess_time = 0
    while True:
        middle = calc_middle(low_number, high_number)
        guess_time += 1
        if middle == right:
            break
        elif middle > right:
            high_number = middle - 1  # 避免无限循环
        else:
            low_number = middle + 1  # 避免无限循环
    return guess_time, right

# 调用函数并打印结果
guess_time, right = guess(0, 100000000)
print(f"我猜了{guess_time}次，正确的数字是{right}")"""


"""import random

def calc_middle(low_number, high_number):
    return int((low_number + high_number) / 2)

def guess(low_number, high_number):
    right = random.randint(low_number, high_number)  # 在函数内部生成随机数
    guess_time = 0
    while True:
        middle = calc_middle(low_number, high_number)
        guess_time += 1
        if middle == right:
            break
        elif middle > right:
            high_number = middle - 1  # 避免无限循环
        else:
            low_number = middle + 1  # 避免无限循环
    return guess_time, right

# 调用函数并处理返回值
guess_time, right = guess(0, 100000000)
print(f"我猜了{guess_time}次，正确的数字是{right}")"""




"""检查违禁词
def check_unsafe_words(word):

    if 'tmd' in word:
        return "********"

if check_unsafe_words("你tm好水"):#等价于FALSE，下面一行不执行
    print("safe word")
comment = "你tm好水"
print(check_unsafe_words(comment))"""

"""理解局部变量与全局变量"""
"""y=5
def func_a():
    global x
    x=10
    y=10
    print(y)
    return x,y
func_a()
print(x)
print(y)"""






"""bmi计算函数（不完整）"""
"""def calc_bmi(height,weight):
    bmi=weight/(height)**2
    return bmi

bmi=calc_bmi(1.7,67.2)
print(bmi)"""


"""def create_game(user_name,password,initial_payment):
   创建一个新游戏注册界面，其中有用户名、密码、首充金额三个内容
   print(f"我的名字是{user_name}，密码是{password}，首充金额为{initial_payment}元")

create_game(password="12341234",initial_payment="0",user_name="又菜又爱玩")"""



"""def result_function():
    print("函数执行完成！")

def function(x,y,result_function):
    print("调用function函数")
    result=x+y
    print(f"答案是{result}")
    result_function()
    return result

function(1,2,result_function)"""



"""my_list=["起床","吃饭","打扰宝哥","睡觉","在宝哥身上睡觉",[1,2,3]]"""
"""another_list=["苑神启动"]
my_list[2]="在宝哥身上睡觉"""
"""print(my_list[2])
print(my_list.index("在宝哥身上睡觉"))
print(my_list.count("在宝哥身上睡觉"))
print(len(my_list))
print(my_list[-1][1])
my_list.extend(another_list)
print(my_list)
my_list.append("不玩苑神导致的")
print(my_list)"""
"""my_list.pop(2)
print(my_list)
my_list.remove("在宝哥身上睡觉")
print(my_list)
del my_list[3]
print(my_list)"""
"""index=0
for item in my_list:
    print(f"第{index}下标的元素是{item}")
    index+=1"""









































