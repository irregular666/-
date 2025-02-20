"""import requests #抓包buff上商品价格

response=requests.get(url='https://buff.163.com/api/index/popular_sell_order?_=1731814008592')

goods_infos=(response.json()['data']["goods_infos"])

for key,value in goods_infos.items():
    print(f"{value['name']}的价格是{value['steam_price_cny']}元")"""

"""import requests #抓包百度

response=requests.get(url='https://www.baidu.com')

print(response.text)"""


"""from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
from selenium.webdriver.chrome.service import Service



#初始化
driver=webdriver.chrome()

service=Service()"""



