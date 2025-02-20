"""import twostudy.exam_two

animal1=twostudy.exam_two.Animal("老虎","肉",3,"汪汪汪",True)

animal1.make_sound()"""

import os

def convert_to_relative_path(absolute_path, base_path):
    # 将绝对路径转换为相对路径
    relative_path = os.path.relpath(absolute_path, base_path)
    return relative_path

# 示例
absolute_path = "C:/Users/31715/Desktop/二轮招新/第一题.md"
base_path = "C:/Users/31715"

relative_path = convert_to_relative_path(absolute_path, base_path)
print("相对路径:", relative_path)

