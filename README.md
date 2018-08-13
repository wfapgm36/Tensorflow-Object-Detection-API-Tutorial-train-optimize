# Tensorflow-Object-Detection-API-Tutorial-train-optimize
[Tensorflow Object Dectection API](https://github.com/tensorflow/models/tree/master/research/object_detection)를 사용해 특정 객체 검출에 특화된 Detection 모델을 만들 수 있는 Tutorial입니다. 


# Summary
우리는 종종 높은 정확도를 조금 양보하더라도 **빠르고(fast) 가벼운(light) Detection 모델**을 원합니다. 또한 수십 개의 객체를 검출하기 보단, **특정 객체 몇 개를 정확하고 빠르게 검출**하기를 원합니다. 예를 들어 도로위의 차량과 사람, 신호등만을 검출하길 원하거나 신용 카드의 숫자를 검출하는 모델을 원하는 경우입니다.

이 튜토리얼은 Tensorflow Object Detection API를 사용해 사용자가 원하는 특정 객체를 검출하는 빠르고 가벼운 모델을 만들 수 있도록 도와줍니다. 그럼에도 높은 정확도를 유지할 수 있는 방법도 알려줍니다.

본 튜토리얼은 신용 카드 숫자 검출 모델 생성을 예시로 설명을 진행합니다.


**개발 환경**
* Ubuntu 16.04
* Python 3.6.6
* Tensorflow 1.9.0
* cuda 9.0

# Table of contents
1. [설치(Install)](#Install)
2. [데이터셋(Dataset) 구성](#Dataset)
>* Step 1. 학습 데이터 준비
>* Step 2. 데이터 라벨링
>* Step 3. csv 파일 생성
>* Step 4. TF Record 파일 생성
>* Step 5. label.pbtxt 파일 생성
3. [학습(Training)](#Train)
4. [모델 구동(Running)](#Running)
5. [결과(Result)](#Result)
6. [Extras](#Extras)

## 설치<a name="Install"></a>

[Tensorflow Object Dectection API](https://github.com/tensorflow/models/tree/master/research/object_detection)을 적당한 폴더에 git clone하고 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)를 참고하여 설치를 완료합니다. ( COCO API installation은 하지 않아도 튜토리얼을 진행하는데 문제는 없습니다. )

## 데이터셋 구성<a name="Dataset"></a>

### Step 1. 학습 데이터 준비

학습에 필요한 이미지 데이터를 준비한다.  관련 이미지는 Google 검색을 통해 다운로드하거나, 가지고 있는 이미지 파일을 사용하면 된다.
검출하고자 하는 객체가 이미지 데이터에 존재해야 한다. 또한 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)에서 알 수 있듯이 학습 시 사용할 이미지 데이터는 RGB image로 jpeg나 png 포멧이어야 한다.

### Step 2. 데이터 라벨링

### Step 3. csv 파일 생성

### Step 4. TF Record 파일 생성

### Step 5. label.pbtxt 파일 생성


## 학습<a name="Train"></a>

## 모델 구동<a name="Running"></a>

## 결과<a name="Result"></a>

## Extras<a name="Extras"></a>
