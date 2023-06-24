import datetime
import logging
import os
import numpy as np
import torch
from PIL import Image
from deepface import DeepFace
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


import logging

# 创建Logger对象
logger = logging.getLogger(__name__)

# 设置日志级别为INFO
logger.setLevel(logging.INFO)

# 创建文件处理器，将日志写入文件
file_handler = logging.FileHandler('pretrain.log')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器，将日志打印到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义日志消息格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 将格式应用到处理器
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到Logger对象
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 训练模型
dataset = "temp_train/train"
txtPath = "temp_list_train.txt"
output_dir="saving"

last_dir=os.path.join(output_dir,'checkpoint-last')
if not os.path.exists(last_dir):
    os.makedirs(last_dir)

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # 使用 'sparse' 因为 labels 是整数形式
    shuffle=True  # 打乱数据顺序
)
# 获取类别标签
labels = train_generator.class_indices
logger.info(labels)

# 创建模型并进行编译
model = DeepFace.build_model("VGG-Face")
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 训练模型
epochs = 6
defined_epochs=epochs
e=3
while epochs>0:
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=e
    )

    # # 获取当前日期和时间
    # current_datetime = datetime.now()
    # print(current_datetime)
    #
    # # 格式化当前日期和时间
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # print(formatted_datetime)
    out_dir=os.path.join(output_dir,'checkpoint-'+str(defined_epochs-epochs))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save(os.path.join(out_dir, "optimizer.h5"))
    model.save(os.path.join(last_dir, "optimizer.h5"))
    logger.info("Saving optimizer to %s", os.path.join(out_dir, "optimizer.h5"))
    epochs-=e

#
# with open(txtPath, "r") as file:
#     lines = file.readlines()
# image_paths = []
# labels = []
# print('成功读取路径')
# for line in lines:
#     line = line.strip()
#     label, image_path = line.split("/")
#     fin_path = dataset + "/" + label + "/" + image_path
#     image_paths.append(fin_path)
#     labels.append(int(label[1:]))
#
# images = []
# for image_path in image_paths:
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     image = np.array(image)
#     images.append(image)
#
# images = np.array(images)
# labels = np.array(labels)
# images = images / 255.0
# print('成功保存图片及对应标签')
# model = DeepFace.build_model("VGG-Face")
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print('成功编译DeepFace自带的VGG-Face模型')
# model.fit(images, labels, batch_size=32, epochs=30)
# model.save("trained_model.h5")
# print('成功训练并保存模型')

# 测试模型
dataset = "temp_test/test"
txtPath = "temp_list_test.txt"
with open(txtPath, "r") as file:
    lines = file.readlines()
image_paths = []
labels = []
for line in lines:
    line = line.strip()
    label, image_path = line.split("/")
    fin_path = dataset + "/" + label + "/" + image_path
    image_paths.append(fin_path)
    labels.append(int(label[1:]))

images = []
for image_path in image_paths:
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    images.append(image)

images = np.array(images)
labels = np.array(labels)
images = images / 255.0

model = load_model(os.path.join(last_dir, "optimizer.h5"))
test_loss, test_accuracy = model.evaluate(images, labels)
print("test_accuracy and loss:")
print(test_accuracy)
print(test_loss)
