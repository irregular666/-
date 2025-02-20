import os
import random


def extract_20_percent_labels_per_folder(cat_dirs, eval_labels_file, labels_dir):
    """
    从每个类别的标签文件夹中随机抽取 20% 的标签数据，并将其写入 eval_labels.txt。
    :param cat_dirs: 包含类别子文件夹的列表
    :param eval_labels_file: 用于保存 eval 标签的文件路径
    :param labels_dir: 标签文件所在目录
    """
    with open(eval_labels_file, 'w') as eval_file:
        for cat_dir in cat_dirs:
            folder_path = os.path.join(labels_dir, cat_dir)  # 构建类别文件夹的路径
            # 检查文件夹是否存在
            if os.path.exists(folder_path):
                # 获取当前类别文件夹中的所有标签文件
                label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

                # 随机抽取 20% 的标签文件
                eval_size = int(len(label_files) * 0.2)
                eval_label_files = random.sample(label_files, eval_size)

                for label_file in eval_label_files:
                    # 获取标签文件的完整路径
                    label_path = os.path.join(folder_path, label_file)

                    # 读取标签文件内容
                    with open(label_path, 'r') as f:
                        # 读取文件的每行数据，这些是19个坐标点
                        coords = f.read().strip().split()  # 获取 19 个点的坐标

                        # 确保每个标签文件包含 19 个点 (38 个数)
                        if len(coords) == 38:
                            coords_str = ' '.join(coords)  # 格式化为空格分隔的字符串
                            eval_file.write(f"{coords_str}\n")  # 将坐标写入 eval_labels.txt
            else:
                print(f"目录不存在: {folder_path}")

    print(f"Successfully extracted 20% of labels from each category to {eval_labels_file}")


def main():
    # 设置路径
    cat_dirs = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03', 'CAT_04', 'CAT_05']  # 所有子文件夹
    eval_labels_file = "C:/Users/31715/Desktop/工作室/cats/val/eval_labels.txt"  # 输出文件
    labels_dir = "C:/Users/31715/Desktop/工作室/cats/cats"  # 标签文件所在目录

    # 抽取并保存 20% 的标签数据到 eval_labels.txt
    extract_20_percent_labels_per_folder(cat_dirs, eval_labels_file, labels_dir)


if __name__ == '__main__':
    main()

