from decimal import Decimal

from decimal import Decimal

money = Decimal(input("请输入您充值的金额："))

# 直接乘以1.15
money *= Decimal('1.15')

# 使用 round() 函数确保结果为整数
money = round(money)

print("您现在卡中的余额为：", money)


