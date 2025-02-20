"""def func_a(x):#这是一个计算2x+1的程序
    y=2*x+1
    return y

x=int(input("请输入x的值："))
y=func_a(x)
print(y)"""


"""import random #抽奖程序

# 初始化集合，包含1到10000的数字
my_set = set(range(1, 10001))

while len(my_set) > 1:
    # 随机选择集合中的一个元素
    chosen_number = random.choice(tuple(my_set))
    # 删除选中的数字
    my_set.remove(chosen_number)
    # 打印选中的数字（可选，但在大量数字时可能不推荐）
    # print(f"选中的数字是：{chosen_number}")

# 最后剩下的一个元素就是“中奖”数字
print(f"最终中奖数字是：{my_set.pop()}")"""

"""set_1={1,2,3,3,3,3,2,2,2,4,5,6,3,3,3,3}#数列去重
list_1=list(set_1)
print(list_1)"""

#字典嵌套
"""score_dict={
    "张三":{
        "数学":400,
        "英语":500,
        "总分":"******",



    },
    "苑神":{"数学":500,
            "英语":600,
            "总分":"keyerror"





            },

    "李四":400,
    "Trump":666
    }
print(sorted(score_dict,reverse=False))"""


#异常捕获

"""def divide(a,b):
    return a/b

def divide_twice(a,b,operator):
    try:
       result=divide(a,b)
       return result
    except Exception as e:
       print(f"不是哥们。别输入0，错误类型是{e}")

divide_twice(10,0,divide)"""

import torch
class Animal:
    def __init__(self,name,diet,age,sound,love_status=False):

        self.name=name
        self.diet=diet
        self.age=age
        self.sound=sound
        self.__love_status=love_status


    def describe(self):

        des=f"名字：{self.name}，食物：{self.diet}，年龄：{self.age},声音：{self.sound}"
        print(des)

    def make_sound(self):
        sound=f"{self.name}发出{self.sound}的声音"
        print(sound)

    def __str__(self):
        return f"大家好，这是一只{self.name},它可以发出{self.sound}的叫声"

    def __add__(self, other):
        if self.__love_status:
            print("我已经有配偶了")
            return ""
        else:
            return f"{self.name}和{other.name}在一起了,他们的孩子叫{self.name[0]}{other.name[0]}"

    def __call__(self):
        self.describe()
        self.make_sound()
        return f""
dog=Animal("老虎","肉",3,"汪汪汪",True)

tiger=Animal("狗","肉",5,"喵喵喵",True)

chicken=Animal("烧鸡","flyfire",19,"那是fsj的siren",True)

print(chicken)






