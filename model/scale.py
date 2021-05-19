import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import horovod.torch as hvd
import sync_batch_norm as sbn
import numpy as np



class Write(nn.Module):
    def __init__(self, msg):
        super(Write, self).__init__()
        self.msg = msg
    def forward(self, x):
        print("Write: ", self.msg, x.size(), flush=True)
        return x

class BottleNeck(nn.Module):
    def __init__(self, n, running=False):
        super(BottleNeck, self).__init__()

        n2 = n//2
        self.block = nn.Sequential(
            sbn.BatchNorm(n, running),
            nn.Conv3d(n, n2, kernel_size=1, bias=False),
            sbn.BatchNorm(n2, running),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv3d(n2, n2, kernel_size=3, padding=1, padding_mode="replicate", bias=False),
            sbn.BatchNorm(n2, running),
            nn.ReLU(inplace=True),
            nn.Conv3d(n2, n, kernel_size=1))

    def forward(self, x):
        res = x
        x = self.block(x)
        x += res
        return x


class Advc(nn.Module):
    def __init__(self, dx, dz, nv, nitr):
        super(Advc, self).__init__()

        self.dx = dx
        self.dz = dz
        ary = []
        for i in range(nitr):
            ary.append(
                nn.Sequential(
                    Write(1),
                    nn.Conv3d(nv, nv*3, kernel_size=3, groups=nv, padding=1, padding_mode="replicate"),
                    Write(2),
                    nn.ReLU(inplace=True),
                    Write(3),
                    nn.Conv3d(nv*3, nv*3, kernel_size=3, groups=nv, padding=1, padding_mode="replicate"),
                    Write(4),
                    nn.ReLU(inplace=True),
                    Write(5),
                    nn.Conv3d(nv*3, nv*3, kernel_size=3, groups=nv, padding=1, padding_mode="replicate"))
            )
        self.blocks = nn.ModuleList(ary)

    def forward(self, p, uvw, i):
        b,n,z,y,x = p.size()
        print("cnv", i, flush=True)
        p = self.blocks[i](p)
        #print("expand", flush=True)
        dz = self.dz.expand(b,-1,-1,-1)
        a = []
        u = uvw[:,0,:,:,:] / self.dx
        v = uvw[:,1,:,:,:] / self.dx
        w = uvw[:,2,:,:,:] / dz
        for nn in range(n):
            pp =  p[:,nn*3  ,:,:,:] * u
            pp += p[:,nn*3+1,:,:,:] * v
            pp += p[:,nn*3+2,:,:,:] * w
            pp = pp.view(b,1,z,y,x)
            a.append(pp)
        p = torch.cat(a, dim=1)
        
        return p

class Phys(nn.Module):
    def __init__(self, nv, nv2, nl, running):
        super(Phys, self).__init__()
        self.phy_d0 = nn.Sequential(
            sbn.BatchNorm(nv+nv2, running=running),
            nn.Conv3d(nv+nv2, nl, kernel_size=1))
        self.phy_d1u = BottleNeck(nl, running=running)
        self.phy_d2u = BottleNeck(nl, running=running)
        self.phy_d3 = BottleNeck(nl, running=running)
        self.phy_d2 = BottleNeck(nl, running=running)
        self.phy_d1 = BottleNeck(nl, running=running)

        self.phy_w = nn.Sequential(
            sbn.BatchNorm(nl, running=running),
            nn.ReLU(inplace=True),
            nn.Conv3d(nl, nl*nv, kernel_size=1, bias=False),
            sbn.BatchNorm(nl*nv, running=running),
            nn.ReLU(inplace=True),
            nn.Conv3d(nl*nv, nl*nv, kernel_size=1, bias=False),
            sbn.BatchNorm(nl*nv, running=running),
            nn.ReLU(inplace=True),
            nn.Conv3d(nl*nv, nl*nv, kernel_size=1),
            nn.Sigmoid())

    def forward(self, v):
        v0 = self.phy_d0(v)

        b,l,z,y,x = v0.size()

        v1u = self.phy_d1u(v0)
        v1d = F.max_pool3d(v0, kernel_size=2)
        v2u = self.phy_d2u(v1d)
        v2d = F.max_pool3d(v1d, kernel_size=2)
        v3 = self.phy_d3(v2d)
        v2 = F.interpolate(v3, size=(z//2,y//2,x//2)) + v2u
        v2 = self.phy_d2(v2)
        v1 = F.interpolate(v2, size=(z,y,x)) + v1u
        v = self.phy_d1(v1)

        w = self.phy_w(v) # B,N,L,Z,Y,X
        v = w .view(b,-1,l,z,y,x) * v.view(b,1,l,z,y,x)
        v = v.sum(2)

        return v


class Net2(nn.Module):

    def __init__(self, vars, nv, nv2, dx, dz, nbuf, qmin, mean, stddev, stddev2):
        super(Net2, self).__init__()

        self.nitr = 2

        self.nv = nv
        self.nv2 = nv2

        dz = torch.from_numpy(dz)

        self.nbuf = nbuf

        self.qlogmin = {}
        self.rvar = {}
        qlogmin = np.empty((1), dtype=np.float32)
        qlogmin[0] = qmin
        qlogmin = np.log(qlogmin)
        for n in range(nv):
            var = vars[n]
            if var[0] == "Q" and var != "QV":
                tmp = ( qlogmin - mean[var] ) / stddev[var]
                self.qlogmin[var] = tmp[0]
            self.rvar[var] = ( stddev[var] / stddev2[var] )**2

        nl = 6
        running = True

        self.advc = Advc(dx, dz, nv, self.nitr)
        self.phys = Phys(nv, nv2, nl, running)

    def forward(self, x):
                
        phy = self.phys(x)
        vs = x[:,0:self.nv,:,:,:]
        for i in range(self.nitr):
            uvw = vs[:,0:3,:,:,:]
            advc = self.advc(vs, uvw, i)
            dv = advc + phy
            vs = vs + dv

        return vs



    def get_loss(self, obs, tend, dat, var, ofact=1.0):
        nbuf = self.nbuf
        obs = obs[:,:,nbuf:-nbuf,nbuf:-nbuf]
        tend = tend[:,:,nbuf:-nbuf,nbuf:-nbuf]
        dat = dat[:,:,nbuf:-nbuf,nbuf:-nbuf]
        if var[0] == "Q" and var != "QV":
            qlogmin = self.qlogmin[var]
            mask = torch.logical_and(obs.gt(qlogmin), dat.gt(qlogmin))
            obs = obs[mask]
            dat = dat[mask]
        loss = self.criterion(dat, tend)
        loss *= self.rvar[var]

        return loss
