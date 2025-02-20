import random




i=random.randint(0,1000)
guess_time=0




while True:
    print("你所输入的数字是:")
    guess=input()
    guess=int(guess)
    guess_time += 1

    if guess<i:
       print("你所输入的数字太小了")
    elif guess>i:
       print("你所输入的数字太大了")
    else:
       print(f"恭喜你猜对了，正确答案是{i}")

       print(f"你一共猜了{guess_time}次")
       break





