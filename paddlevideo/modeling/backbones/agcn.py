import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.initializer as init
from ..registry import BACKBONES


def einsum(x, A):#'ncuv,nctv->nctu'
    
    x = x.transpose((0, 1, 3, 2))
    y = paddle.matmul(A, x)
    return y


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node)) # 每个中心节点 靠近重心的邻居集，邻接矩阵，看列索引为中心节点
    Out = normalize_digraph(edge2mat(outward, num_node)) # 每个中心节点 远离重心的邻居集，邻接矩阵
    A = np.stack((I, In, Out)) # axis=0, return shape: 3, V, V
    return A

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                    (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                    (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                    (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                    (21, 14), (19, 14), (20, 19)]
inward = inward_ori_index # 外源点->内目标点
outward = [(j, i) for (i, j) in inward] # 内->外
neighbor = inward + outward

def get_bone(joint_data):
    bone_data = paddle.zeros_like(joint_data)
    for v1, v2 in inward_ori_index:
        bone_data[:, :, :, v1, :] = joint_data[:, :, :, v1, :]-joint_data[:, :, :, v2, :]
    return bone_data

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


class TemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size*dilation-dilation)//2
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1),)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2, 3, 4], residual=True, residual_kernel_size=1):
        super(MultiScale_TemporalConv, self).__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=branch_channels, kernel_size=1, padding=0,),
                nn.BatchNorm2D(branch_channels),
                nn.ReLU(),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=branch_channels, kernel_size=1, padding=0,),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2D(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=branch_channels, kernel_size=1, padding=0, stride=(stride, 1), ),
            nn.BatchNorm2D(branch_channels)
        ))

        if not residual:
            self.residual = lambda x:0
        elif (in_channels==out_channels) and (stride==1):
            self.residual = lambda x:x
        else:
            self.residual = TemporalConv(in_channels=in_channels, out_channels=out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = paddle.concat(branch_outs, axis=1)
        out += res
        return out


class CTRGC(nn.Layer):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 2 or in_channels == 9 or in_channels==3:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2D(self.in_channels, self.rel_channels, kernel_size=1,)
        self.conv2 = nn.Conv2D(self.in_channels, self.rel_channels, kernel_size=1,)
        self.conv3 = nn.Conv2D(self.in_channels, self.out_channels, kernel_size=1,)
        self.conv4 = nn.Conv2D(self.rel_channels, self.out_channels, kernel_size=1,)
        self.tanh = nn.Tanh()

    def forward(self, x, A=None, alpha=1):
        
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x) 
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = einsum(x1, x3)#'ncuv,nctv->nctu'
        return x1


class unit_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), )

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.LayerList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1,),
                    nn.BatchNorm2D(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = paddle.create_parameter(shape=A.shape, default_initializer=init.Assign(A.astype(np.float32)), dtype=paddle.float32)
            ##
            #self.PA.stop_gradient = True
            ##
        else:
            self.A = paddle.to_tensor(A, dtype='float32', stop_gradient=True)
        self.alpha_test = nn.ParameterList()
        for i in range(self.num_subset):
            self.alpha_test.append(paddle.create_parameter(shape=(1,), default_initializer=init.Constant(value=0.0), dtype=paddle.float32))
        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha_test[i])
            y = z + y if y is not None else z
        y = self.bn(y)

        y += self.down(x)
        y = self.relu(y)
        return y

class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=9, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

######实则为CTRGCN的实现！！！##########

@BACKBONES.register()
class AGCN(nn.Layer):
    def __init__(self, num_class=30, num_point=25, num_person=1, in_channels=2,
                 drop_out=0, adaptive=True):
        super(AGCN, self).__init__()

        self.graph = Graph(**{'labeling_mode': 'spatial'})

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        ##1028
        drop_out = 0.5
        ##
        if drop_out>0.:
            self.drop_out = nn.Dropout2D(drop_out)
        else:
            self.drop_out = lambda x: x
        
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def forward(self, x):

        N, C, T, V, M = x.shape

        x = x.transpose((0, 4, 3, 1, 2))
        x = x.reshape((N, M * V * C, T))
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        x = x.reshape((N, M, V, C, T))
        x = x.transpose((0, 1, 3, 4, 2))
        x = x.reshape((N * M, C, T, V))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1
        ####1028
        x = self.drop_out(x)
        ####

        return x