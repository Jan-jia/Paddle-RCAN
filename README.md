﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿# 论文复现：[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://paperswithcode.com/paper/image-super-resolution-using-very-deep)

*****

* RCAN
  * [一、简介](#一简介)
  * [二、复现结果](#二复现结果)
  * [三、数据集](#三数据集)
  * [四、环境依赖](#四环境依赖)
  * [五、预训练模型](#五预训练模型)
  * [六、快速开始](#六快速开始)
    * [训练](#训练)
    * [测试](#测试)
    * [评价指标](#评价指标)
  * [七、模型信息](#七模型信息)

# **一、简介**

***

本项目基于paddlepaddle框架复现Residual Channel Attention Networks(RCAN).RCAN网络是一种Residual in Residua（RIR）结构来形成的非常深的网络，它由几个具有长跳跃连接的残差组组成。每个残差组包含一些具有短跳过连接的残差块。同时，RIR允许通过多个跳转连接绕过丰富的低频信息，使主网络专注于学习高频信息。并加入了通道注意机制，通过考虑通道之间的相互依赖性，自适应地重新缩放通道特征。

#### **论文**

Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." *Proceedings of the European conference on computer vision (ECCV)*. 2018.

#### **参考项目**

https://github.com/yulunzhang/RCAN

#### **项目aistudio地址**

https://aistudio.baidu.com/aistudio/projectdetail/3000561
版本为Commit_3，运行`main.ipynb`即可开始训练。
# 二、复现结果

#### **指标（在set14上测试）**

|        模型        | PSNR  |  SSIM  |
| :----------------: | :---: | :----: |
|        论文        | 28.98 | 0.7910 |
|     Paddle训练     | 28.99 | 0.7913 |
| 预训练模型权重转换 | 28.98 | 0.7910 |

# **三、数据集**

训练集下载： [DIV2K dataset](https://aistudio.baidu.com/aistudio/datasetdetail/110995) ，解压到 `data/` 文件夹中。训练所需的文件夹为`DIV2K/DIV2K_train_HR`和`DIV2K/DIV2K_train_LR_bicubic`。

测试集已整理好在 `data/benchmark/` 和 `Test_code/` 中 。

# **四、环境依赖**

硬件：GPU、CPU

框架：PaddlePaddle >=2.2.0

# **五、预训练模型**

下载地址：[百度网盘](https://pan.baidu.com/s/1ODQVPD2TPLO7cYnZa--9yw)，提取码：hiu0

将所有预训练模型存放在 `experiment/model/` 中

其中 `RCAN_BIX2.pdparams` 用于训练，`model_6.pdparams` 为基于paddlepaddle训练出来的模型，`RCAN_BIX4.pt` 为原作者提供的预训练模型，`RCAN_BIX4.pdparams` 为转换原作者提供的预训练模型。

预训练模型的验证及结果见 `Inference`，分别直接运行两个文件夹中的 `main.py` 即可得到 numpy 的 seed=1 时随机生成输入的结果。

运行 `Inference/val.py` 即可得到输出结果相差的验证。

# **六、快速开始**

### **训练**

原作者提供的训练脚本中，采用 `RCAN_BIX2.pt` 作为初始权重使用，我们参考作者的做法，将他提供的预训练权重转为paddle的权重 `RCAN_BIX2.pdparams` 并开始训练。

```
cd Train_code
LOG=./../experiment/RCAN_BIX4_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt

python main.py --model RCAN --save RCAN_BIX4_G10R20P48 --data_test Set14 --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --reset --chop --save_models --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pdparams 2>&1 | tee $LOG
```

训练结果和日志见 `experiment/RCAN_BIX4_G10R20P48/`

由于训练中断过，我们分了两次训练，其中后缀为 `_1` 表示第一次训练的日志，后缀为`_2` 表示第二次训练的日志

### **测试**

```
cd Test_code/code

# 测试我们自己训练出来的模型
python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_threads 1 --n_resblocks 20 --n_feats 64 --pre_train ../../experiment/model/model_6.pdparams --test_only --save_results --chop --self_ensemble --save RCANplus --testpath ../LR/LRBI --testset Set14

# 测试作者提供的预训练模型(已转换为paddle的权重)
python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_threads 1 --n_resblocks 20 --n_feats 64 --pre_train ../../experiment/model/RCAN_BIX4.pdparams --test_only --save_results --chop --self_ensemble --save RCANplus --testpath ../LR/LRBI --testset Set14
```

测试结果将保存在 `Test_code/SR/BI/RCANplus/Set14/x4` 中。

### **评价指标**

运行 `Test_code/` 下的代码 `Evaluate_PSNR_SSIM.m` 即可得到PSNR和SSIM的结果 。

# **七、模型信息**

模型的总体信息如下：

| 信息     | 说明         |
| -------- | ------------ |
| 框架版本 | Paddle 2.2.0 |
| 应用场景 | 图像生成     |
| 支持硬件 | GPU / CPU    |

