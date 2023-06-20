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
file_handler = logging.FileHandler('ORL_92x112/result/eval2.log')
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



models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace","ArcFace"]
# model_name = models[0]
metrics = ["cosine", "euclidean", "euclidean_l2"]
# metric_name = metrics[0]


def eval(result, name):
    # logger.info('===========================')
    # logger.info(result)
    isMatched=False
    corrects=0
    for path in result:
        print(path)
        name_match=path[0].split('/')[-2]
        if name_match==name:
            isMatched=True
            print(name_match)
            corrects+=1
    return  corrects,isMatched

def detect(img_path, db_path, model, metric):
    # logger.info(img_path)
    recognition = DeepFace.find(img_path=img_path, db_path=db_path, model_name=model, distance_metric=metric, enforce_detection=False)

    try:
        result = np.array(recognition[0])
    except Exception:
        logger.error(recognition[0])
        logger.error(recognition[0].shape)
    return result

if __name__ == '__main__':
    db_path = 'ORL_92x112/train'
    test_path = 'ORL_92x112/test'

    # grid_precision = [[0, 0, 0] for _ in range(10)]
    # grid_recall = [[0, 0, 0] for _ in range(10)]

    for i in range(6,len(models)):
        model = models[i]
        for j in range(len(metrics)):
            metric = metrics[j]
            logger.info('--------------------------------')

            corrects = 0
            expected=0
            total=0
            acc=0


            for dir in os.listdir(test_path):
                # logger.info('--------------------------------')
                print('expected+=%s',len(os.listdir(test_path + '/' + dir)))
                expected += len(os.listdir(test_path + '/' + dir))
                img = os.listdir(test_path + '/' + dir)[0]
                res = detect(test_path + '/' + dir + '/' + img, db_path, model, metric)
                total+=len(res)
                print('dir=%s',dir)
                correct,isMatched=eval(res, dir)
                corrects+=correct
                if isMatched==True:
                    acc+=1


            precision=corrects / total
            recall=corrects / expected
            # 这里准确度指的是 只根据最高匹配度匹配到的人来判断是否match，match则acc++
            accuracy=acc/40
            logger.info(models[i])
            logger.info(metrics[j])
            logger.info('precision=%f',precision)
            logger.info('recall=%f', recall)
            logger.info('accuracy=%f',accuracy)


    # logger.info('grid_precision')
    # logger.info(grid_precision)
    # logger.info('grid_recall')
    # logger.info(grid_recall)

    # txt = open('result.txt', 'w')
    # txt.write('grid_precision')
    # txt.write(str(grid_precision))
    # txt.write('grid_recall')
    # txt.write(str(grid_recall))
    # txt.close()