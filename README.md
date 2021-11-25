# AIStudio-2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第3名方案

## 项目描述
本项目是[基于飞桨实现花样滑冰选手骨骼点动作识别大赛](https://aistudio.baidu.com/aistudio/competition/detail/115)：花样滑冰选手的细粒度骨骼点动作识别大赛B榜第三名方案。本项目基于ICCV2021论文[CTRGCN](https://arxiv.org/abs/2107.12213)和[Focal loss](https://arxiv.org/abs/1708.02002)、[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)套件构建节点流、骨骼流、节点运动流、骨骼运动流四流框架进行动作识别，取得了B榜第三，A榜15的成绩。本项目是从AI Studio项目迁移来，为了方便一键fork和项目复现，请前往已公开的项目：[2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第3名方案](https://aistudio.baidu.com/aistudio/projectdetail/3000114?contributionType=1)

## 项目结构
```
-|data
-|configs
-|docs
-|paddlevideo
-|tools
-README.MD
-2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第3名方案.ipynb
-2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第3名方案.md
-__init__.py
-main.py
-requirements.txt
-setup.py
-run.sh
-submission_A.csv
-submission_B.csv
```
### 注意：
用于集成的五个模型文件在output/AGCN下，因github上传容量限制，没传上来，在这里提供百度网盘链接：链接：https://pan.baidu.com/s/1CEme1aXNDBZkEkowYWXxEw 提取码：zpc6 , 下载后放在当前目录下即可

## 使用方式 
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/3000114?contributionType=1)  
B：查看文档2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第3名方案.ipynb或2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第3名方案.md即可知晓项目使用方式

最后，祝您使用愉快！如果有bug，欢迎在AI Studio项目评论区留言~
