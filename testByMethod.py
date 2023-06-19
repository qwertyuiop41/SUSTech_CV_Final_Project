import logging
import warnings
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace

logging.basicConfig(filename='face_detection.log')
dataset = [
    ["dataset/img1.jpg", "dataset/img2.jpg", True],
    ["dataset/img5.jpg", "dataset/img6.jpg", True],
    ["dataset/img6.jpg", "dataset/img7.jpg", True],
    ["dataset/img8.jpg", "dataset/img9.jpg", True],
    ["dataset/img1.jpg", "dataset/img11.jpg", True],
    ["dataset/img2.jpg", "dataset/img11.jpg", True],
    ["dataset/img1.jpg", "dataset/img3.jpg", False],
    ["dataset/img2.jpg", "dataset/img3.jpg", False],
    ["dataset/img6.jpg", "dataset/img8.jpg", False],
    ["dataset/img6.jpg", "dataset/img9.jpg", False],
]


models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
# model_name = models[0]
metrics = ["cosine", "euclidean", "euclidean_l2"]
# metric_name = metrics[0]
detectors = ["opencv", "mtcnn"]
db_path="/Users/siyiwang/Desktop/wsy/CV/deepface/tests/dataset"

def evaluate(condition):
    global num_cases, succeed_cases

    if condition:
        succeed_cases += 1

    num_cases += 1


def verifyTest():
    global succeed_cases, num_cases
    print("不同face detectors下的人脸验证(是否是同一个人): DeepFace.verify")
    for detector in detectors:
        print(detector + " face detector")
        for i in range(0, len(dataset)):
            res = DeepFace.verify(dataset[i][0], dataset[i][1], detector_backend=detector, model_name="VGG-Face")
            assert isinstance(res, dict)
            assert "verified" in res.keys()
            assert res["verified"] in [True, False]
            assert "distance" in res.keys()
            assert "threshold" in res.keys()
            assert "model" in res.keys()
            assert "detector_backend" in res.keys()
            assert "similarity_metric" in res.keys()
            assert "facial_areas" in res.keys()
            assert "img1" in res["facial_areas"].keys()
            assert "img2" in res["facial_areas"].keys()
            assert "x" in res["facial_areas"]["img1"].keys()
            assert "y" in res["facial_areas"]["img1"].keys()
            assert "w" in res["facial_areas"]["img1"].keys()
            assert "h" in res["facial_areas"]["img1"].keys()
            assert "x" in res["facial_areas"]["img2"].keys()
            assert "y" in res["facial_areas"]["img2"].keys()
            assert "w" in res["facial_areas"]["img2"].keys()
            assert "h" in res["facial_areas"]["img2"].keys()
            print("case : " + str(i) + str(res["verified"] == dataset[i][2]))
            evaluate(res["verified"] == dataset[i][2])
        print(str(detector + " success rate:" + str(succeed_cases / num_cases)))
        succeed_cases = 0
        num_cases = 0

    print("----------------------------------------------------------------------------------------------")


def analysisTest():
    succeed_cases = 0
    num_cases = 0
    print("Facial analysis test.人脸属性分析(age,gender,race,emotion)")
    for index in range(1, 51):
        img = "dataset/img" + str(index) + ".jpg"
        print(img)
        demography_objs = DeepFace.analyze(img, ["age", "gender", "race", "emotion"])
        for demography in demography_objs:
            print("Age: ", demography["age"])
            print("Gender: ", demography["dominant_gender"])
            print("Race: ", demography["dominant_race"])
            print("Emotion: ", demography["dominant_emotion"])

            evaluate(demography.get("age") is not None)
            evaluate(demography.get("dominant_gender") is not None)
            evaluate(demography.get("dominant_race") is not None)
            evaluate(demography.get("dominant_emotion") is not None)

    print("成功检测的比例=" + str(float(succeed_cases / num_cases)))
    print("----------------------------------------------------------------------------------------------")


def findTest(img_path):
    print("从数据集中检索当前人脸相似的图片:DeepFace.find")
    dfs = DeepFace.find(img_path="dataset/img22.jpg", db_path="dataset")
    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        print(df.head())
        evaluate(df.shape[0] > 0)

    # print('-----------')
    # for line in dfs_np[0]:
    #     print(line)
    #     match_path=line[0]
    #     result = DeepFace.verify(img1_path=img_path, img2_path=match_path)
    #     print(result)
    #     print(result['verified'])

    return dfs
    # print(succeed_cases / num_cases)


def representTest():
    print("该函数用于将面部图像表示为特征向量DeepFace.represent")
    print("强制检测是否有人脸存在 的参数：enforce_detection")
    black_img = np.zeros([224, 224, 3])
    try:
        DeepFace.represent(img_path=black_img)
        exception_thrown = False
    except:
        exception_thrown = True

    if exception_thrown:
        print("强制开启人脸是否存在检测成功")
    assert exception_thrown is True

    # -------------------------------------------

    # enforce detection off for represent
    try:
        objs = DeepFace.represent(img_path=black_img, enforce_detection=False)
        exception_thrown = False

        # validate response of represent function
        assert isinstance(objs, list)
        assert len(objs) > 0
        assert isinstance(objs[0], dict)
        assert "embedding" in objs[0].keys()
        assert "facial_area" in objs[0].keys()
        assert isinstance(objs[0]["facial_area"], dict)
        assert "x" in objs[0]["facial_area"].keys()
        assert "y" in objs[0]["facial_area"].keys()
        assert "w" in objs[0]["facial_area"].keys()
        assert "h" in objs[0]["facial_area"].keys()
        assert isinstance(objs[0]["embedding"], list)
        assert len(objs[0]["embedding"]) == 2622  # embedding of VGG-Face
    except Exception as err:
        print(f"Unexpected exception thrown: {str(err)}")
        exception_thrown = True

    if not exception_thrown:
        print("不检测人脸是否存在")
    assert exception_thrown is False

    for i in range(1,51):
        result = DeepFace.represent(img_path="dataset/img"+str(i)+".jpg", model_name=models[0])
        print("特征维度为：{}".format(len(result)))


# verifyTest()
# analysisTest()
findTest(img_path="/Users/siyiwang/Desktop/wsy/CV/deepface/tests/dataset/img22.jpg")
# representTest()




