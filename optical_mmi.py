import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalMMIConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 n_bits=6, wavelength_spacing=10.0):
        """
        光学相变材料卷积层
        Args:
            n_bits: 权重量化位数（论文中使用6-bit精度）
            wavelength_spacing: 波长间隔(nm)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.n_bits = n_bits
        self.wavelength_spacing = wavelength_spacing

        # 可训练权重参数（实际物理实现通过相变材料状态）
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        # 偏置项（如果使用）
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # 初始化参数
        self.reset_parameters()

        # 注册缓冲区用于量化权重
        self.register_buffer('quant_levels', torch.linspace(-1, 1, 2 ** n_bits))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weights(self):
        """模拟相变材料6-bit量化特性"""
        with torch.no_grad():
            # 将权重归一化到[-1, 1]范围
            abs_max = torch.max(torch.abs(self.weight))
            normalized_weights = self.weight / abs_max

            # 量化到离散级别
            diff = torch.abs(normalized_weights.unsqueeze(-1) - self.quant_levels)
            quantized = self.quant_levels[torch.argmin(diff, dim=-1)]

            return quantized * abs_max
        return self.weight

    def forward(self, x):
        """
        光学卷积前向传播
        模拟多波长输入和非相干叠加
        """
        # 1. 量化权重（模拟相变材料编程）
        quantized_weight = self.quantize_weights()

        # 2. 应用标准卷积（软件模拟光学矩阵乘法）
        x = F.conv2d(
            x,
            quantized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        # 3. 模拟非相干光叠加（直接输出光强）
        return x

    def simulate_optical_propagation(self, input_field):
        """
        高级物理仿真（概念性实现）
        实际硬件中由相变材料结构实现
        """
        # 此处应包含光学传播方程和相变材料相互作用
        # 简化实现：直接使用卷积结果
        return self.forward(input_field)