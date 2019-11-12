## mnist数据集
建立一个pytorch学习的环境。使用torch1.1.0和torchvision0.3.0  
仿照搭建一个mnist识别的神经网络，并完成以下工作：  
1. 更换优化器：从SGD到Adam：发现sgd能够用0.01的学习了，但是Adam就不行，学不到东西，必须把学习率降低，然后能够学到东西。
2. 更换loss函数：从nll_loss转成cross_entropy
3. 更换网络的结构：
4. 如何画出网络结构：常用的方法是使用tensorboard进行可视化，需要安装tensorboard，tensorflow等，具体的见[链接](https://zhuanlan.zhihu.com/p/58961505)， 瞄的发现这个tensorboard一直不太好使  
[代码链接](https://github.com/ASONG0506/pytorch-learning/blob/master/01-classification/mnist/main.py)
----------

## 使用cifat数据集进行数据分类
### 0. 代码实现的总体步骤

**代码使用**：  
```
cd cifar
python cifar.py --lr=0.01
```

1. 实现数据集的加载：
2. 实现不同的网络结构：通常所有的网络实现都放置在一个models的文件夹中
3. 实现网络的损失函数，优化函数的设置
4. 设置网络的训练以及测试流程的函数实现
5. 训练、测试与输出结果的打印
    
### 1. 数据集的加载
 
本部分内容使用的是cifar的数据，故直接使用官方的torchvision中集成好的cifar数据集的格式，其主要的套路如下所示
```
# 1.1 定义transforms，基于torchvision.transforms工具：
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# 1.2. 封装成datasetloader的形式，之后就可以直接调用相应的数据了
trainset = torchvision.datasets.CIFAR10(root = "../../data/", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = "../../data/", train=False, transform=transform_test, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle = False, num_workers = 2)
```
### 2. 网络的构建

1. 这部分的模型不断更新中，详见[仓库链接](https://github.com/ASONG0506/pytorch-learning/tree/master/01-classification/cifar/models/cifar)
2. 具体网络的实现讲解：1
    1. LeNet网络的实现：[链接](https://github.com/ASONG0506/pytorch-learning/blob/master/01-classification/cifar/models/cifar/LeNet.py)
        1. 网络结构比较简单，直接通过构建一个继承于nn.Module的类，实现__init__()函数和forward函数即可。
    2. vgg网络实现：[链接](https://github.com/ASONG0506/pytorch-learning/blob/master/01-classification/cifar/models/cifar/vgg.py)
        1. vgg根据网络结构层数的不同，具有4种结构，根据是否使用bn又可以分为8种不同的小结构，所以通过**函数名**和一个**list构成的配置项cfg**进行不同的网络结构的区分；
        2. 通过make_layers函数来生成特征提取器，使用一个全连接层实现分类器。
        3. 整个VGG的类继承于nn.Module类，需要实现__init__()函数和forward函数，其中__init__()函数负责进行类的一些变量的设置和结构的搭建，forward函数负责前向传播中的各层关系的组织。
        4. nn.Sequential(*layers)通过一个list的网络层关系来组织起来一个神经网络
        
    3. resnet网络实现：[链接](https://github.com/ASONG0506/pytorch-learning/blob/master/01-classification/cifar/models/cifar/resnet.py)
        1. resnet最重要的特点就是采用残差结构来使网络结构得到了极大的加深，具体的网络结构可以参考[博客](https://blog.csdn.net/jing_xian/article/details/78878966)，其中根据不同的网络深度，具有5种结构，两种浅层的网络结构是以**basic block**为基础进行搭建的，结构如图所示：  
        ![basic block](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCE54c21065f14baeea917981547474a0ec/23828)  
        另外三种更深层的网络结构使用的是**Bottleneck block结构**为基础进行构建的，该结构采用1-3-1大小的卷积核实现了在压缩通道中进行3×3卷积，从而有效降低了计算量，且所有的bottleneck中输出的通道数都是输入的通道数的4倍，在采用了stride>1的卷基层后，特征通道数增加，宽高维度下降，因此对于残差结构来说，存在特征的通道维度对齐的问题，这里使用的是1×1的卷积实现，具体结构如图所示:  
        ![bottleneck](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCEcb08976bf25580765b05d89b4ab81db8/23835)  
        各种网络结构具体组成如图:  
        ![resnet architecture](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCE597a80fef7e04e926da2f171d05d8c12/23838)  
        2. 首先构建一个Basic block的类，Bottleneck的类，并在此基础上构建ResNet的类，可以通过不同的配置方式生成不同的网络结构。
    4. googlenet网络实现：[链接](https://github.com/ASONG0506/pytorch-learning/blob/master/01-classification/cifar/models/cifar/googlenet.py)，googlenet网络的详细讲解可参考[链接](https://blog.csdn.net/shuzfan/article/details/50738394)
        1. 网络结构，在同一层中使用不同的卷积核进行卷积计算，网络的基础结构如图所示：  ![原始的googlenet Inception](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCEf5d135f61ac1479ee0057c5ff5e7d320/23937)  
        基于此改进的googlenet Inception结构如图所示，通过1*1卷积进行通道降维，有效减小计算量![google inception v2](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCEfd30434fc99e8ce5b78b4c5cce151c48/23940)  
        网络结构如图所示：  [googlenet图片太长，没有插进来](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCEe7775625a31ea235cc64e84e94f463a2/23945)
        具体的网络参数如图所示： ![googlenet](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCE376369aba2db12248aeb34e0ccb80ce6/23948)
        2. 网络的实现：  
        由于网络的一个inception中包含了多个分支，因此需要输入每一个分支的具体的通道参数，逐个分支进行实现，就ok了，详见代码
    5. mobilenet网络实现：[链接](https://github.com/ASONG0506/pytorch-learning/blob/master/01-classification/cifar/models/cifar/mobilenet.py) 网络的原理讲解可以参考[博客](https://zhuanlan.zhihu.com/p/31551004)               
        1. 结构方面，简单来说，就是通过把普通卷积拆分为depth-wise convolution和point-wise convolution来实现轻量化，从而减小计算量的同时还能保持相当不错的精度。如下图所示：  
        ![mobilenet](https://note.youdao.com/yws/public/resource/fd103c4515462526cceea50e79224b7a/xmlnote/WEBRESOURCE8923c8e33914309c781d7559280ab8f3/24032)
        2. 具体的实现：比较重要的一点就是这个depth-wise的实现方式，**通过在nn.Conv2d()函数中加入一个与输入通道数相同的group参数实现逐层卷积depth-wise**，其余方面的实现比较简单和常规，详见代码
        
    6. mobilenetv2网络实现
    
* 训练与测试函数的构建
* 训练、测试与中间输出结果的打印
* 保存模型文件等