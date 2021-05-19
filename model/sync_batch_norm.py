from horovod.torch.mpi_ops import allreduce_async, synchronize
import horovod.torch as hvd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class BatchNorm(nn.Module):
    def __init__(self, nc: int, running=False):
        super(BatchNorm, self).__init__()
        if hvd.is_initialized() and hvd.size() > 1:
            self.obj = _SyncBatchNorm(nc, running=running)
        else:
            self.obj = nn.BatchNorm3d(nc)

    def forward(self, x):
        return self.obj(x)


class _SyncBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, running=False):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.running = running

    def forward(self, input):
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1

        if ( not self.training and self.track_running_stats ) or hvd.size()==1:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)
        else:
            with torch.no_grad():
                var, mean = torch.var_mean(input, [0,2,3,4], unbiased=False)

            mean_handle = allreduce_async(mean)
            var_handle = allreduce_async(var)
            mean_all = synchronize(mean_handle)
            var_all = synchronize(var_handle)

#            if hvd.rank()==0:
#                print(hvd.rank(), self.num_features, mean.mean(), mean_all.mean(), var.mean(), var_all.mean(), flush=True)

            with torch.no_grad():
                self.running_mean[:] = (1.0 - self.momentum) * self.running_mean + self.momentum * mean_all
                self.running_var[:]  = (1.0 - self.momentum) * self.running_var  + self.momentum * var_all

            if self.running:
                return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
            else:
                return F.batch_norm(input, mean_all, var_all, self.weight, self.bias, False, 0.0, self.eps)

