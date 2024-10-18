import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

'''
    在光流估计和图像处理中，warp 函数的作用是将一个图像（tenInput）根据给定的光流场（tenFlow）进行变换，
    以生成一个新的图像。这个过程可以模拟图像中的像素如何从一个帧移动到另一个帧。

in：
    tenInput: 输入 [batch_size, channels, height, width]。
    tenFlow:  光流场 [batch_size, 2, height, width]，2 表示每个像素在x水平 和 y垂直 方向上的运动。
out：
    tenOutput: 变换后的图像 [batch_size, channels, height, width]。  
'''
def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)

        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

"""
个像素坐标加上它的光流即为该像素点对应在目标图像的坐标.

https://blog.csdn.net/qq_41942564/article/details/108637368
由于取值范围是-1->1   整个单位的距离是2,所以一个网格的长宽距离分别为：w/2,h/2

对于output中的每一个像素(x, y)，它会根据流值在input中找到对应的像素点(x+u, y+v)，并赋予自己对应点的像素值，这便完成了warp操作。但这个对应点的坐标不一定是整数值，因此要用到插值或者使用邻近值，也就是选项mode的作用
但这里一般会将m和n的取值范围归一化到[-1, 1]之间，[-1, -1]表示input左上角的像素的坐标，[1, 1]表示input右下角的像素的坐标
对于超出这个范围的坐标，函数将会根据参数padding_mode的设定进行不同的处理

"""


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    # 假设设备是 CPU
    device = torch.device("cpu")

    # 创建一个随机图像 batch (batch_size=2, 3 channels, 5 height, 5 width)
    tenInput = torch.randn(2, 3, 5, 5, device=device)

    # 创建一个随机光流场 (batch_size=2, 2 flow components, 5 height, 5 width)
    tenFlow = torch.randn(2, 2, 5, 5, device=device)

    # 调用之前定义的 warp 函数
    warped_images = warp(tenInput, tenFlow)

    # 打印输出
    print("Original Image Batch Shape: ", tenInput.shape)
    print("Flow Field Shape: ", tenFlow.shape)
    print("Warped Image Batch Shape: ", warped_images.shape)

    import matplotlib.pyplot as plt

    # 选择第一个图像和对应的光流以及变换后的图像进行可视化
    input_image = tenInput[0].cpu().detach().numpy().transpose(1, 2, 0)  # Convert to HxWxC
    flow_field = tenFlow[0].cpu().detach().numpy().transpose(1, 2, 0)
    warped_image = warped_images[0].cpu().detach().numpy().transpose(1, 2, 0)

    # Plot the original image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title('Original Image')

    # Plot the flow field (u, v)
    plt.subplot(1, 3, 2)
    plt.imshow(flow_field[0], cmap='coolwarm')  # Assume flow field is visualized using the first channel (u)
    plt.colorbar()
    plt.title('Flow Field (U)')

    # Plot the warped image
    plt.subplot(1, 3, 3)
    plt.imshow(warped_image)
    plt.title('Warped Image')
    plt.show()