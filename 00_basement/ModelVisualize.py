# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 
# function : 用于对torch.nn.Module进行理解
# 参考链接： https://blog.csdn.net/qq_27825451/article/details/90550890
# pytorch中的基础是 torch.nn.Module模块的理解
# 主要包括了：如何在构建网络的时候方便合理地设置网络的结构和名称；如何打印网络的结构，如何迭代打印网络分块结构和其对应的名称等等。
# --------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class TestModel1(nn.Module):
    """
    直接将所有的权重的名称在init函数中进行设置.
    """
    def __init__(self, name):
        super(TestModel1, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.bn1= nn.BatchNorm2d(10)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(10, 2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(2)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(2*8*8, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x).flatten()

        x = self.linear1(x)
        x = self.linear2(x)

        return x


class TestModel2(nn.Module):
    """
    将带有权重的层在init中进行设置,没有权重的通过torch.nn.functional函数式变成进行实现.
    """
    def __init__(self, name):
        super(TestModel2, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.bn1= nn.BatchNorm2d(10)
        # self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(10, 2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(2)
        # self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(2*8*8, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.activation1(x)
        x = F.relu(x)  # 这里修改为了F.prelu的函数式编程方式
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.activation2(x)
        x = F.relu(x)  # 这里修改为了F.prelu的函数式编程方式
        x = self.pool2(x).flatten()

        x = self.linear1(x)
        x = self.linear2(x)

        return x


class TestModel3(nn.Module):
    """
    使用nn.sequential对于网络的结构进行包装，将小的结构包装成一个大的块
    """
    def __init__(self, name):
        super(TestModel3, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 10, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense_block = nn.Sequential(
            nn.Linear(10 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(10 * 8 * 8)
        out = self.dense_block(res)
        return out


class TestModel4(nn.Module):
    """
    使用nn.sequential对于网络的结构进行包装，将小的结构包装成一个大的块
    这里加上了各个层的名字的设置
    """
    def __init__(self, name):
        super(TestModel4, self).__init__()
        self.conv_block = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 32, 3, 1, 1)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(2)),
                ("conv2", nn.Conv2d(32, 10, 3, 1, 1)),
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(2))]
            )
        )
        self.dense_block = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(10 * 8 * 8, 128)),
                ("relu3", nn.ReLU()),
                ("linear2", nn.Linear(128, 10))]
            )
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(10 * 8 * 8)
        out = self.dense_block(res)
        return out


class TestModel5(nn.Module):
    """
    使用nn.sequential对于网络的结构进行包装，将小的结构包装成一个大的块
    这里加上了各个层的名字的设置
    """
    def __init__(self, name):
        super(TestModel5, self).__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module("conv1", nn.Conv2d(3, 32, 3, 1, 1))
        self.conv_block.add_module("relu1", nn.ReLU())
        self.conv_block.add_module("pool1", nn.MaxPool2d(2))
        self.conv_block.add_module("conv2", nn.Conv2d(32, 10, 3, 1, 1))
        self.conv_block.add_module("relu2", nn.ReLU())
        self.conv_block.add_module("pool2", nn.MaxPool2d(2))

        self.dense_block = nn.Sequential()
        self.dense_block.add_module("linear1", nn.Linear(10*8*8, 128))
        self.dense_block.add_module("relu3", nn.ReLU())
        self.dense_block.add_module("linear2", nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(-1)
        out = self.dense_block(res)
        return out








if __name__ == "__main__":
    x = torch.ones((1, 3, 32, 32), dtype=torch.float32).to("cuda")
    testModel1 = TestModel1("test model demo 1").to("cuda")
    out = testModel1(x)
    # torch.save(testModel, "./testModelDemo1.pth")
    print(testModel1)
    """
    这里打印的结果:
    TestModel(
      (conv1): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation1): PReLU(num_parameters=1)
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(10, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation2): PReLU(num_parameters=1)
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (linear1): Linear(in_features=128, out_features=32, bias=True)
      (linear2): Linear(in_features=32, out_features=10, bias=True)
    )
    说明:直接打印网络模型.
    好的模型的表示方法是将网络所有带有权重的层,放到init函数中进行初始化,不带有权重的层,放到forward中,表示带有权重的是其固有的属性
    """
    print("*******************************************************")

    # 使用带参数和不带参数的层进行隔离的方式
    testModel2 = TestModel2("test model demo 2").to("cuda")
    out = testModel2(x)
    # torch.save(testModel2, "./testModelDemo2.pth")
    print(testModel2)
    """
    打印结果2:
    TestModel2(
      (conv1): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(10, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (linear1): Linear(in_features=128, out_features=32, bias=True)
      (linear2): Linear(in_features=32, out_features=10, bias=True)
    )
    可以发现:这里打印的模型中,没有激活函数relu了.
    我们可以将模型保存结果testmodeldemo1和testModelDemo2使用netron打开进行查看,发现一个问题,model1显示的结果是正确的,
    但是model2虽然能够正常显示,但是显然和我们搭建的网络实际执行的顺序是不一致的,中间遗漏了relu层,
    说明netron是按照pth的self中'固有属性'的层进行显示的,因此在我们的模型可视化的时候,是不能直接用netron可视化的pth为结果的,因为可能不对.
    
    """
    print("*******************************************************")

    # 使用nn.sequential()对于网络进行分块。
    testModel3 = TestModel3("test model demo 3").to("cuda")
    out = testModel3(x)
    # torch.save(testModel3, "./testModelDemo3.pth")
    print(testModel3)
    """
    输出的结果为如下：
    可以看出来，每一个模块的名称是在__init__()函数中所赋予的名称，其中的每一个子模块可以通过index索引来拿到，
    TestModel3(
      (conv_block): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (dense_block): Sequential(
        (0): Linear(in_features=640, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=10, bias=True)
      )
    )
    """

    print("*******************************************************")
    print("使用nn.sequential 以及使用层名字对每一层进行赋值")
    testModel4 = TestModel4("test model demo 4").to("cuda")
    out = testModel4(x)
    print(testModel4)
    print(out)
    """
    打印的结构如下所示
    可以看到在网络块中已经有了各个层对应的名称，可以直接使用key值进行索引了。
    TestModel4(
      (conv_block): Sequential(
        (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu1): ReLU()
        (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu2): ReLU()
        (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (dense_block): Sequential(
        (linear1): Linear(in_features=640, out_features=128, bias=True)
        (relu3): ReLU()
        (linear2): Linear(in_features=128, out_features=10, bias=True)
      )
    )
    """

    print("*******************************************************")
    print("还可以使用另外一种方式对于sequential中的层赋名字")
    testModel5 = TestModel5("test model demo 5").to("cuda")
    out = testModel5(x)
    print(testModel5)
    print(out)
    """
    打印结果如下，和上一中方法结果一致。
    TestModel5(
      (conv_block): Sequential(
        (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu1): ReLU()
        (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu2): ReLU()
        (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (dense_block): Sequential(
        (linear1): Linear(in_features=640, out_features=128, bias=True)
        (relu3): ReLU()
        (linear2): Linear(in_features=128, out_features=10, bias=True)
      )
    )
    """

    # nn.Modules也通过children(), named_children(), modules(), names_modules()方法提供了对其成员层的访问方式
    # nn.Modules也通过children(), named_children()分别可以打印第一层的submodules
    print("*******************************************************")
    for i in testModel5.children():
        print(i)
        print(type(i))
    """
    输出结果如下，可以看到，是将一层的children的modules打印出来。
    Sequential(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu1): ReLU()
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu2): ReLU()
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    <class 'torch.nn.modules.container.Sequential'>
    Sequential(
      (linear1): Linear(in_features=640, out_features=128, bias=True)
      (relu3): ReLU()
      (linear2): Linear(in_features=128, out_features=10, bias=True)
    )
    <class 'torch.nn.modules.container.Sequential'>
    """

    # nn.Modules也通过children(), named_children(), modules(), names_modules()方法提供了对其成员层的访问方式
    print("*******************************************************")
    # 通过named_children()可以打印出来名称
    for i in testModel5.named_children():
        print(i)
        print(type(i))
    """
    可以发现，打印出来的类型变成了tuple了，因为包括了名称和具体的参数。
    ('conv_block', Sequential(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu1): ReLU()
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu2): ReLU()
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ))
    <class 'tuple'>
    ('dense_block', Sequential(
      (linear1): Linear(in_features=640, out_features=128, bias=True)
      (relu3): ReLU()
      (linear2): Linear(in_features=128, out_features=10, bias=True)
    ))
    <class 'tuple'>
    """

    # nn.Modules也通过children(), named_children(), modules(), names_modules()方法提供了对其成员层的访问方式
    print("*******************************************************")
    # 通过modules()方法来迭代地打印每个层
    for i in testModel5.modules():
        print(i)
        print("-----------------------------")
    """
    TestModel5(
      (conv_block): Sequential(
        (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu1): ReLU()
        (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu2): ReLU()
        (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (dense_block): Sequential(
        (linear1): Linear(in_features=640, out_features=128, bias=True)
        (relu3): ReLU()
        (linear2): Linear(in_features=128, out_features=10, bias=True)
      )
    )
    -----------------------------
    Sequential(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu1): ReLU()
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu2): ReLU()
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    -----------------------------
    Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    -----------------------------
    ReLU()
    -----------------------------
    MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    -----------------------------
    Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    -----------------------------
    ReLU()
    -----------------------------
    MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    -----------------------------
    Sequential(
      (linear1): Linear(in_features=640, out_features=128, bias=True)
      (relu3): ReLU()
      (linear2): Linear(in_features=128, out_features=10, bias=True)
    )
    -----------------------------
    Linear(in_features=640, out_features=128, bias=True)
    -----------------------------
    ReLU()
    -----------------------------
    Linear(in_features=128, out_features=10, bias=True)
    -----------------------------
    """

    # nn.Modules也通过children(), named_children(), modules(), names_modules()方法提供了对其成员层的访问方式
    print("*******************************************************")
    # 通过named_modules()方法来迭代地打印每个层
    for i in testModel5.named_modules():
        print(i)
        print("-----------------------------")
    """
    打印结果如下所示，迭代打印了每一个modules的名称以及其内容。
    ('', TestModel5(
      (conv_block): Sequential(
        (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu1): ReLU()
        (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu2): ReLU()
        (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (dense_block): Sequential(
        (linear1): Linear(in_features=640, out_features=128, bias=True)
        (relu3): ReLU()
        (linear2): Linear(in_features=128, out_features=10, bias=True)
      )
    ))
    -----------------------------
    ('conv_block', Sequential(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu1): ReLU()
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu2): ReLU()
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ))
    -----------------------------
    ('conv_block.conv1', Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    -----------------------------
    ('conv_block.relu1', ReLU())
    -----------------------------
    ('conv_block.pool1', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
    -----------------------------
    ('conv_block.conv2', Conv2d(32, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    -----------------------------
    ('conv_block.relu2', ReLU())
    -----------------------------
    ('conv_block.pool2', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
    -----------------------------
    ('dense_block', Sequential(
      (linear1): Linear(in_features=640, out_features=128, bias=True)
      (relu3): ReLU()
      (linear2): Linear(in_features=128, out_features=10, bias=True)
    ))
    -----------------------------
    ('dense_block.linear1', Linear(in_features=640, out_features=128, bias=True))
    -----------------------------
    ('dense_block.relu3', ReLU())
    -----------------------------
    ('dense_block.linear2', Linear(in_features=128, out_features=10, bias=True))
    -----------------------------
    
    """




