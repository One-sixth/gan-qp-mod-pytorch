import torch
import torch.nn.functional as F


class leaky_twice_relu(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        """
        :type x: torch.Tensor
        """
        x = torch.where(x > 1, 1 + 0.1 * (x - 1), x)
        x = torch.where(x < 0, 0.1 * x, x)
        return x


class TwiceLog(torch.jit.ScriptModule):
    __constants__ = ['scale']
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    # 第一种实现
    # 不使用torch.sign，因为当x为0时，sign为0，此时梯度也为0
    # x为0时，torch.abs的梯度也为0，所以下面表达式不使用
    # sign = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, -1))
    # x = torch.log(torch.abs(x)+1) * sign
    # 第二种实现，当x=负数，而目标为正数时，梯度无效，原因，使用where后，图像是连接在一起，但导数函数仍然是分开的，例子 x=-1，x-3=0.7
    # x = torch.where(x >= 0, torch.log(x + 1), -1 * torch.log(torch.abs(x - 1)))
    # 第三种实现，当前实现，全程可导，而且导数域一致，忘记x本身就是线性可导了
    @torch.jit.script_method
    def forward(self, x):
        """
        :type x: torch.Tensor
        """
        x = torch.where(x != 0, torch.log(torch.abs(x)+1) * torch.sign(x), x)
        x = x * self.scale
        return x


class TanhScale(torch.jit.ScriptModule):
    __constants__ = ['scale']
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    @torch.jit.script_method
    def forward(self, x):
        """
        :type x: torch.Tensor
        """
        x = torch.tanh(x) * self.scale
        return x


if __name__ == '__main__':
    a = torch.empty(50).normal_()
    a.requires_grad = True
    opt = torch.optim.SGD([a], lr=0.0001)
    act = TwiceLog()
    while True:
        bk = a.clone()
        b=torch.empty(50).normal_()
        # b.require_grad = True
        c=(act(a)-b).abs().mean()
        c.backward()
        opt.step()
        if torch.isnan(a).sum().item() != 0:
            print('Found nan')
