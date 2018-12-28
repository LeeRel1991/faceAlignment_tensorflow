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

|网络|stage|300W-common| 300W-challege |AFLWW2000 |speed(ms/face)
|:---|:---|:---|:---|:---|:---
dan_vgg_112_300WAugment(paper)| -/2 | -/4.42 | -/4.57 |  - | -
dan_vgg_112_300W| 1/2 | -| - | - | - | -
dan_vgg_112_300WAugment| 1/2 | 5.16/4.82 | 10.08/9.64 |  22.67/23.68 | 3-5
dan_mobilenet_112_300WAugment| 1/2 | 6.97/5.29| 12.37/9.65 | 24.19/24.35
prnet_256_300WLP(paper)| - | 7.47| 14.99 | 6.30 | 10 | 75


* MSE normalized by diagonal box distance

|网络|stage|300W-common| 300W-challege |AFLWW2000 |speed(ms/face)
|:---|:---|:---|:---|:---|:---
dan_vgg_112_300WAugment(paper)| 1/2 | -/1.35 | -/2.00 |  - | -
dan_vgg_112_300WAugment| 1/2 | 1.56/1.45 | 2.62/2.48 |  4.49/4.37 | 5/20
dan_mobilenet_112_300WAugment| 1/2 | 2.11/2.09| 3.10/3.19 | 4.91/4.90
prnet_256_300WLP(paper)| - |2.22 |  3.67 | 2.3 | 10
FAN_256_300WLP(paper)| - |- |  - | 3.38 | 10
FAN_256_300WLP| - |2.15 |  3.68 | 2.57 | 25