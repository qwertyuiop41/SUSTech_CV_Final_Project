import os
import re

import numpy as np
from deepface import DeepFace

import logging

# 创建Logger对象
logger = logging.getLogger(__name__)

# 设置日志级别为INFO
logger.setLevel(logging.INFO)

# 创建文件处理器，将日志写入文件
file_handler = logging.FileHandler('eval.log')
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



models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
# model_name = models[0]
metrics = ["cosine", "euclidean", "euclidean_l2"]
# metric_name = metrics[0]


def eval(result, name):
    # logger.info('===========================')
    # logger.info(result)
    corrects=0
    total=len(result)
    for path in result:
        name_match=path[0].split('/')[-2]
        if name_match==name:
            corrects+=1
    return  corrects,total

def detect(img_path, db_path, model, metric):
    logger.info(img_path)
    recognition = DeepFace.find(img_path=img_path, db_path=db_path, model_name=model, distance_metric=metric, enforce_detection=False)
    try:
        result = np.array(recognition[0])
    except Exception:
        logger.info(recognition[0])
        logger.info(recognition[0].shape)
    return result

if __name__ == '__main__':
    db_path = 'CVDataset/train'
    test_path = 'CVDataset/test'

    grid_precision = [[0, 0, 0] for _ in range(8)]
    grid_recall = [[0, 0, 0] for _ in range(8)]

    for i in range(len(models)):
        model = models[i]
        for j in range(len(metrics)):
            metric = metrics[j]

            corrects = 0
            expected=0
            total=0

            for dir in os.listdir(test_path):
                logger.info('--------------------------------')
                expected += len(os.listdir(test_path + '/' + dir))
                img = os.listdir(test_path + '/' + dir)[0]
                res = detect(test_path + '/' + dir + '/' + img, db_path, model, metric)
                correct,match=eval(res, dir)
                corrects+=correct
                total+=match


            grid_precision[i][j] = corrects / total
            grid_recall[i][j] = corrects / expected
            logger.info(models[i])
            logger.info(metrics[j])
            logger.info('precision=%f',corrects / total)
            logger.info('recall=%f', corrects / expected)


    logger.info('grid_precision')
    logger.info(grid_precision)
    logger.info('grid_recall')
    logger.info(grid_recall)

    txt = open('result.txt', 'w')
    txt.write('grid_precision')
    txt.write(str(grid_precision))
    txt.write('grid_recall')
    txt.write(str(grid_recall))
    txt.close()