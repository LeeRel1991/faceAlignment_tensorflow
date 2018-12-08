## Facial Alignment
__faceAlignment_tensorflow__ 基于tensorflow 实现了现有先进的人脸对齐算法(the state-of-the-art deep learning models)
主要包括以下几个方面:
1. 利用tensorfow复现了现有人脸对齐算法的训练代码，尽量达到paper中的水平
2. 基于numpy实现了人脸对齐领域常用的几种评价指标(如MSE, normalized MSE, etc.)
3. 基于tf.data API 实现了对不同标注格式的人脸对齐数据集的加载，预处理和可视化
4. 简单的数据增广

#### 依赖
* tensorflow >=1.4.0
* python == 3.5
* opencv == 3.3.0
* numpy

#### 组织结构
* face_alignment: 根目录
    * model_zoo: 定义算法网络模型文件
    * tools: 训练，评测,数据可视化，数据增广脚本
    * utils：辅助功能，如绘制点/矩形框，数据集加载，日志，评测指标等
* model: 训练模型存放路经
* data 相关数据资源文件，如meanshape等
* README.md

#### 支持的算法
* [DAN](https://arxiv.org/pdf/1706.01789.pdf)

* [FAN](https://www.adrianbulat.com/)

* [PRNet](https://arxiv.org/pdf/1803.07835.pdf)


#### 支持的数据集
* 300W (image-pts pair)
* 300W-LP(image-mat pair)
* AFLW2000-3D(image-mat pair)
* 300VW(todo)
#### 性能对比

* MSE normalized by pupil distance

|网络|stage|300W-common| 300W-challege |AFLWW2000 |speed(ms/face) | 显存占用(%)
|:---|:---|:---|:---|:---|:---|:---
dan_vgg_112_300W| 1/2 | 1| 14 | 14 | 1.3 | 65
dan_vgg_112_300WAugment| 1/2 | 7.43/6.10 | 14.99/11.40 |  49.013/40.23 | 3-5 |25
dan_resnet_112_300WAugment| 1/2 | 8.933/0.06133| 13.430/11.345 | 44.712/44.072 | 4-7 |
dan_mobilenet_112_300WAugment| 1/2 | 8.323/8.294| 14.892/14.617 |48.134/47.506 | 2-5 |
prnet_256_300WLP| - | 7.607| 14.631 | 14.988 | 10 | 75
dan_resnet_112_300WAugment| 1/2 | 1| 32 | 8 | 1.3 | 78

* MSE normalized by diagonal box distance

|网络|stage|300W-common| 300W-challege |AFLWW2000 |speed(ms/face) | 显存占用(%)
|:---|:---|:---|:---|:---|:---|:---
prnet_256_300WLP(paper)| - |2.22 |  3.67 | 2.3 | 10 | -
FAN_256_300WLP(paper)| - |- |  - | 3.38 | 10 | -
FAN_256_300WLP(train)| - |2.15 |  3.68 | 2.57 | 25 | -