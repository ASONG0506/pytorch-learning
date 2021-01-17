# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 
# function : 用于对torch.nn.Module进行理解
# 参考链接： https://blog.csdn.net/qq_27825451/article/details/90550890
# pytorch中的基础是 torch.nn.Module模块:
# --------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    torch.save(testModel2, "./testModelDemo2.pth")
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
    # 使用nn.sequential()对于网络进行分块。
    



