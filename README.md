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
>* Step 1. [학습 데이터 준비](#Preparedata)
>* Step 2. [데이터 라벨링](#Datalabelling)
>* Step 3. [csv 파일 통합](#Mergecsv)
>* Step 4. [TF Record 파일 생성](#Maketfrecord)
>* Step 5. [object-detection.pbtxt 파일 생성](#Makepbtxt)
3. [학습(Training)](#Train)
4. [모델테스트(Running)](#Running)
5. [결과(Result)](#Result)
6. [Extras](#Extras)

## 1. 설치<a name="Install"></a>

[Tensorflow Object Dectection API](https://github.com/tensorflow/models/tree/master/research/object_detection)을 적당한 폴더에 git clone하고 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)를 참고하여 설치를 완료합니다. ( COCO API installation은 하지 않아도 튜토리얼을 진행하는데 문제는 없습니다. )

**본 튜토리얼의 대부분의 작업 및 명령어는 models/research/object_detection에서 실행됩니다.** 또한 아래와 같은 디렉토리 구조를 만들어 아래와 같은 순서대로 진행합니다.
* 데이터셋 구성
* 모델 학습
* 모델 구동

![directory](./docs/img/directory.png)

## 2. 데이터셋 구성<a name="Dataset"></a>

### Step 1. 학습 데이터 준비<a name="Preparedata"></a>

학습에 필요한 이미지 데이터를 준비하는 과정입니다. object_detection/training 폴더를 만들어 폴더에 학습에 필요한 이미지를 Google 검색을 통해 다운로드하거나, 가지고 있는 이미지 파일을 저장해 줍니다. 단, 이미지에는 검출하고자 하는 객체가 존재해야 합니다. 또한 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)에서 알 수 있듯이 학습 시 사용할 이미지 데이터는 **RGB image로 jpeg나 png 포맷**이어야 합니다.

본 튜토리얼에서 사용한 데이터셋은 **.png 파일과 .csv 파일이 한 쌍 씩** 이루고 있는데, 여기서 **.png 파일은 학습할 이미지 데이터**이며, **.csv 파일에는 이미지 안에 존재하는 객체에 대한 정보**( class, x, y, width, height )가 저장되어 있습니다. 

> **.csv 파일의 생성에 대한 설명은 Step 2. 데이터 라벨링 목록에서 이어서 하게 됩니다.**

---
![file_list](./docs/img/file_list.png)

* 학습에 사용할 데이터셋 리스트
---
![card_img](./docs/img/card_img.png)

* 학습에 사용할 카드 이미지
---
![csv_format](./docs/img/csv_format.png)

* 학습에 사용할 데이터셋의 csv 포맷( 왼쪽부터 객체의 **class(숫자), x, y, width, height** )
---


### Step 2. 데이터 라벨링<a name="Datalabelling"></a>

다음으로 Object Detection 모델을 학습하기 위해서는 이미지 데이터에 존재하는 객체에 **Labelling**을 하는 과정이 필요합니다. 이와 관련한 많은 프로그램이 존재하지만 이 튜토리얼에선 github에 open source로 공개되어 있는 [LabelImg](https://github.com/tzutalin/labelImg)를 사용하길 추천합니다.

본 튜토리얼에서 제공받았다고 가정한 객체 정보는 .csv 파일로 존재하지만, 위 프로그램을 사용하여 라벨링을 진행했을 경우 결과는 .xml 형식으로 얻게됩니다. 이는 [datitran](https://github.com/datitran/raccoon_dataset)의 github에 있는 xml_to_csv.py 소스를 사용해 간단하게 하나의 .csv 파일로 변경 가능합니다.

만약 본 튜토리얼처럼 라벨링 결과가 class, x, y, width, height의 순서대로 정의된 .csv 파일로 얻게된다면 아래 **Step 3. csv 파일 통합** 코드를 이용해서 수 많은 .csv 파일을 하나로 통합할 수 있습니다.

결과적으로 라벨링한 후 얻게되는 오브젝트의 정보를 갖고있는 output이 .csv 또는 .xml인 것은 중요하지 않고 이를 이용해 이미지와 객체 정보들을 하나의 TFRecord 파일로 변경하기 위해 정해진 포맷에 맞춰 하나의 .csv 파일로 통합하는 것이 중요합니다

### Step 3. csv 파일 통합<a name="Mergecsv"></a>

xml_to_csv.py 소스를 사용해서 .xml 파일을 .csv파일로 만들었다면 약간의 소스 코드 수정을 통해 TFRecord 포맷에 맞는 하나의 통합된 .csv을 얻게 되었을 것 입니다. 그러나 본 튜토리얼 데이터셋처럼 각 이미지 파일마다 .csv 파일을 얻게 될 경우, TFRecord 파일을 생성하기 위해서 이를 하나의 통합된 .csv 파일로 만들어줘야 합니다.

통합하는 과정에서 TFRecord 파일을 생성할 때 요구되는 포맷을 맞춰줘야 하는데, 이는 다음과 같습니다.
>( filename, width, height, class, xmin, ymin, xmax, ymax ) 

**여기서 width와 height는 image의 사이즈이고, Labelling을 통해 얻은 width와 height는 bounding box의 사이즈이므로 혼동하지 않아야 합니다.** 본 튜토리얼에서 제공하는 [소스코드](./docs/code/merge_csv.ipynb)를 통해 간단히 .csv 파일들을 포맷에 맞게 통합할 수 있습니다. 통합된 모습은 다음과 같습니다.

![merged_csv](./docs/img/merged_csv.png)


### Step 4. TFRecord 파일 생성<a name="Maketfrecord"></a>

Object Detection 모델을 학습시킬 때 마다 이미지와 .csv 파일을 한 쌍으로 데이터를 보관하고 이용하는 것은 비효율적이고 관리하기에도 좋지 않습니다. Tensorflow Object Detection API는 이를 해결하기 위해 이미지와 .csv 파일은 **TFRecord**라는 하나의 파일로 만드는 방법을 사용했습니다.

object_detection/data 폴더에 통합된 train_labels.csv 을 넣어줍니다. 그 후 아래 명령어를 실행해서 TFRecord 파일을 생성합니다.
> python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

이 과정은 제공되는 [소스코드](./docs/code/generate_tfrecord.py)를 object_detection 폴더에 넣은 후 사용하면 됩니다. 단, 아래와 같이 사용자 데이터에 적합하게 class와 path를 수정해야합니다. 
---
![TFRecord_class](./docs/img/TFRecord_class.png)

![TFRecord_path](./docs/img/TFRecord_path.png)
---
data 폴더에 생성된 tfrecord 파일은 다음과 같습니다.

![tfrecord](./docs/img/tfrecord.png)

TFRecord에 대한 더 자세한 설명은 [여기](http://bcho.tistory.com/1190)를 참고하길 바랍니다.

### Step 5. object-detection.pbtxt 파일 생성<a name="Makepbtxt"></a>

본 튜토리얼에서는 카드 숫자 검출을 위해 0 ~ 9의 10가지 숫자를 검출해야할 class로 지정했습니다. TFRecord 파일은 .pb 포으로 학습 시 데이터를 읽어 오는데, 여기서 객체 정보에 대한 label도 .pbtxt형식으로 읽게 됩니다. 따라서 앞서 생성한 object_detection/training 폴더에 [이것](./docs/code/object-detection.pbtxt)과 같은 object-detection.pbtxt 파일을 만들어줘야 합니다. 모델 테스트를 위하여 같은 파일을 object_detection/data 폴더에도 복사하여 넣어줍니다. 총 2개의 같은 .pbtxt 파일이 생성되었습니다.


## 3. 학습<a name="Train"></a>

앞의 과정을 통해서 학습에 필요한 데이터를 수집하고, 라벨링하고, TFRecord로 변환하는 과정을 통해 모델을 학습할 준비가 되었습니다. 학습에 앞서 Tensorflow Object Detection API는 사용자가 간단히 학습 환경을 변경할 수 있도록 object_detection/samples/configs에 .config 파일을 여러 개 준비해두었습니다. 

우리는 좀 더 빠르고 정확한 모델을 학습시키기 위하여 SSD(Single Shot Detector)와 MobileNet_V2를 사용하겠습니다. 이를 위해서 [ssd_mobilenet_v2_coco.config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config) 파일을 복사하여 training 폴더에 넣어주도록 합시다.

이 때, 모델을 처음부터 학습시키기 위해서는 매우 많은 시간이 필요하게 되므로 COCO dataset으로 미리 학습된 모델(Pretrained Model)을 사용해 우리 모델을 Transfer learning을 시키도록 합시다. 또한 우리가 data 폴더에 생성한 train.record와 object-detection.pbtxt을 사용하여 모델을 학습시키기 위해 .config 파일을 아래와 같이 수정합니다. .config 파일에 대한 자세한 내용은 Extras를 참고하세요.
---
>9 ~~num_classes: 90~~
>>9 num_classes: 10


>156 ~~fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"~~
>>156 fine_tune_checkpoint: "ssd_mobilenet_v2_coco_2018_03_29/model.ckpt"


>175 ~~input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100"~~
>>175  input_path: "data/train.record"


>177 ~~label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"~~
>>177 label_map_path: "data/object-detection.pbtxt"
---

Pretrained 모델 다운로드와 각 모델간의 속도 비교는 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)에서 할 수 있습니다. ssd_mobilenet_v2_coco 모델을 다운로드 받아 object_detection 폴더에 넣도록 합시다. 그 후 object_detection 디렉토리에서 아래와 같은 명령을 실행하여 학습을 시작합니다.
> python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_coco.config

학습이 정상적으로 시작되면 아래와 같이 동작합니다.

![train](./docs/img/loss.png)

각 step마다 loss 값이 출력되고 잘 학습되는 경우 loss는 0에 수렴합니다. 적당히 loss가 0 ~ 2 정도로 수렴한다 싶으면 학습을 종료하고 모델을 테스트해봅니다.

Tensorflow Object Detection API에서 제공하는 .config 파일을 사용해 본 후 카드 번호 검출에 최적화한 .config 파일을 사용해 모델을 학습시켜보고 결과를 비교해보고 싶다면 [이것](./docs/code/ssd_mobilenet_v2_coco.config)을 사용해 보시길 바랍니다. 

## 4. 모델 테스트<a name="Running"></a>

모델 테스트를 위하여 다음과 같은 과정이 필요합니다.
1. 추론 그래프 추출
2. 추론 그래프를 사용하여 객체 검출

우선 추론 그래프 추출을 위해 학습을 종료하면 training 폴더에 많은 파일들이 생긴 것을 확인할 수 있습니다. 이를 객체 검출에 사용하기 위해 추론 그래프를 추출하려면 다음 명령어를 object_detection 폴더에서 실행합니다.
> python3 export_inference_graph.py \
 --input_type image_tensor \
 --pipeline_config_path training/ssd_mobilenet_v2_coco.config \
 --trained_checkpoint_prefix training/model.ckpt-xxxxx \
 --output_directory num_recognition
  
model.ckpt-xxxxx의 xxxxx부분에 저장된 모델 번호를 쓰고 명령어를 실행하면 num_recognition 폴더가 생성되고 폴더안에 frozen_inference_graph.pb(추론 그래프)가 생성된 것을 알 수 있습니다.

추론 그래프 추출이 완료되었으면 학습된 모델을 사용하여 객체를 추출하기 위해 object_detection/object_detection_tutorial.ipynb을 다음과 같이 수정합니다. 
---
![model_name](./docs/img/model_name.png)
---
추론 모델 추출 결과 num_recognition 파일이 생성되었기 때문에 model_name을 num_recognition으로 변경하고 PAATH_TO_CKPT도 바뀐 모델 경로로 변경합니다. PATH_TO_LABELS는 레이블에 사용되는 문자용 목록으로, 앞서 정의한 object_detection.pbtxt로 변경합니다. 마지막으로 여기서 정의한 클래스는 10개(0 ~ 9)이므로 NUM_CLASSES의 값을 수정합니다.  
---
![test_images](./docs/img/test_images.png)
---
테스트를 하기 위해 테스트할 이미지를 object_detection/test_images 폴더에 업로드합니다. 이미지들을 test_{} 형식에 맞춰 업로드하면 range의 범위에 따라 순서대로 이미지를 불러옵니다.

업로드 후 object_detection_tutorial.ipynb를 실행하면 test_images 디렉토리의 이미지에 대해 num_recognition/frozen_inference_graph.pb을 사용하여 객체를 검출(추론)하고 그 결과를 출력합니다.

---
![convert_rgb](./docs/img/convert_rgb.png)
---
본 튜토리얼에서 사용한 카드가 있는 test_images의 이미지들은 gray-scale로 1channel인 반면에 모델을 학습시키기 위해서는 3채널로 넣어야 하기 때문에 강제로 3채널의 RGB 이미지 파일로 변환해 줍니다. 



## 5. 결과<a name="Result"></a>
![result](./docs/img/result.png)
![result](./docs/img/result2.png)

결과를 보면 학습이 잘 되어 카드 번호 검출이 잘 이루어지는 것을 알 수 있다. 
## 6. Extras<a name="Extras"></a>
