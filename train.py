from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import horovod.torch as hvd


import math
import sys
import time
import os
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("--pth")
parser.add_argument("--step")
args = parser.parse_args()

path = args.path


debug = args.test
#debug = False

pth = args.pth

step = args.step


import numpy as np
import random
import netCDF4
import net2

if debug:
    torch.autograd.set_detect_anomaly(True)
    import warnings
    warnings.filterwarnings('error')

hvd.init()
rank_size = hvd.size()
myrank = hvd.rank()


if pth:
    if not os.path.exists(pth):
        if myrank==0:
            print(f"{pth} is not exists")
        exit()


if myrank==0:
    if debug:
        print("debug mode")
    print(f"rank_size: {rank_size}")
    print(f"# of threads: {torch.get_num_threads()}")
    if pth:
        print(f"pth file: {pth}")


torch.set_num_threads( torch.get_num_threads() - 1 )





if debug:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    

year = 2020
month = 8
day = 26
hour0 = 11
if debug:
    hour1 = 12
    #hour1 = 16
    #exps = ["mean", "mdet"]
    exps = ["mean"]
    nt = 9
    #nt = 121
    max_epoch = 10
    #max_epoch = 3
    #max_epoch = 1
    nint = 1
    neval = max_epoch
    batch_size = 2
else:
    hour1 = 16
    exps = ["mean", "mdet"]
    nt = 121
#    max_epoch = 2000
    max_epoch = 1000
    nint = 10
    neval = 10
#    neval = 20
    batch_size = 2


nt1 = nt - 1





P0 = 100000.0

Rdry = 287.04
CPdry = 1004.64
Rvap = 461.50
CPvap = 1846.0
CVdry = CPdry - Rdry
CVvap = CPvap - Rvap
CL = 4218.0
CI = 2106.0

nexp = len( exps )
#vars = ["DENS", "MOMX", "MOMY", "MOMZ", "RHOT", "QV", "QC", "QR", "QI", "QS", "QG"]
vars = ["DENS", "MOMX", "MOMY", "MOMZ", "QV", "QC", "QR", "QI", "QS", "QG", "RHOT"]
nvar = len( vars )

ne = nvar + 2






nx = 256
ny = 256
nz = 60
nbuf = 10




nhour_t = hour1 - hour0
ndata_t = nt1 * nhour_t * nexp
ndata_ts = ndata_t // rank_size



ndata_e = nt1 * nexp







batch_ts = batch_size
#batch_ts = batch_size // rank_size

batch_num = ndata_ts // batch_ts

if myrank==0:
    print(f"nt1: {nt1}, nhour_t: {nhour_t}, nexp: {nexp}, ndata_t: {ndata_t}, ndata_ts: {ndata_ts}, batch_size: {batch_size}, batch_ts: {batch_ts}, batch_num: {batch_num}, max_epoch: {max_epoch}", flush=True)

if ndata_ts * rank_size != ndata_t:
    if myrank==0:
        print(f"size of process is invalid: {nt1}, {hour1-hour0}, {nexp}, {rank_size}")
    exit()

#if batch_ts * rank_size != batch_size:
#    if myrank==0:
#        print(f"batch_size is invalid: {batch_size}, {rank_size}")
#    exit()

if batch_num * batch_ts != ndata_ts:
    if myrank==0:
        print(f"batch_size is invalid: {batch_num}, {batch_ts}, {ndata_ts}")
    exit()


fname_t = []
fname_e = []
hour_t = []
hour_e = []
for exp in exps:
    for hour in range(hour0, hour1):
        fname = "%s/%04d%02d%02d%02d00_%s/history.nc"%(path,year,month,day,hour,exp)
        fname_t.append(fname)
        hour_t.append(hour)
    fname = "%s/%04d%02d%02d%02d00_%s/history.nc"%(path,year,month,day,hour1,exp)
    fname_e.append(fname)
    hour_e.append(hour1)


nfile_t = len(fname_t)
assert nfile_t == nexp * nhour_t



if myrank==0:
    print("read mean and stddev", flush=True)
    start = time.time()


#mean = np.random.randn(nvar,nz,ny,nx)
#stddev = np.ones((nvar,nz,ny,nx))

#fname = f"{path}/mean.nc"
#fname = f"{path}/mean2.nc"
#fname = f"{path}/mean3.nc"
#fname = f"{path}/mean4.nc"
#fname = f"{path}/mean5.nc"
#fname = f"{path}/mean6.nc"
fname = f"{path}/mean7.nc"
file = netCDF4.Dataset(fname)
mean = {}
stddev = {}
stddev2 = {}
for nv in range(nvar):
    var = vars[nv]
    var2 = var
    if var=="MOMX":
        var2 = "U"
    if var=="MOMY":
        var2 = "V"
    if var=="MOMZ":
        var2 = "W"
    if var=="RHOT":
        var2 = "T"
    mean[var] = file.variables[var2+"_mean"][:]
    stddev[var] = file.variables[var2+"_stddev"][:]
    stddev2[var] = file.variables[var2+"_dt_stddev"][:]
file.close()

if myrank==0:
    print(f"end (%.2f s)"%(time.time() - start), flush=True)



fname = "dz.nc"
file = netCDF4.Dataset(fname)
dz = file.variables["dz"][:]
file.close





qmin = 1e-7
patt = re.compile('\d{12}_\w{4}')
def get_var(fname, hour, it0, it1):
    nt = it1 - it0

#    data = np.random.randn(nt,nvar,nz,ny,nx).astype(np.float32)

    data = np.empty((nt,ne,nz,ny,nx), dtype=np.float32)

    file = netCDF4.Dataset(fname)
    for nv in range(nvar):
        var = vars[nv]
        v = file.variables[var]
        d = v[it0:it1,:,:,:]
        _,iz,iy,ix = d.shape
        d = d[:,iz-nz:,iy-ny:,ix-nx:]
        if var == "DENS":
            dens = d.copy()
            d = np.log(d)
        if var == "MOMX":
            d[:,:,:,1:] = ( d[:,:,:,:nx-1] + d[:,:,:,1:] ) * 0.5
            d = d / dens
        if var == "MOMY":
            d[:,:,1:,:] = ( d[:,:,:ny-1,:] + d[:,:,1:,:] ) * 0.5
            d = d / dens
        if var == "MOMZ":
            d[:,1:,:,:] = ( d[:,:nz-1,:,:] + d[:,1:,:,:] ) * 0.5
            d = d / dens
        if var == "QV":
            r = d * Rvap
            cv = d * CVvap
            qdry = 1.0 - d
            d = np.log(d)
        if var[0] == "Q" and var != "QV":
            if var == "QC" or var == "QR":
                cv = cv + d * CL
                qdry = qdry - d
            if var == "QI" or var == "QS" or var == "QG":
                cv = cv + d * CI
                qdry = qdry - d
            d[d<qmin] = qmin
            d = np.log(d)
        if var == "RHOT":
            pt = d / dens
            r = r + qdry * Rdry
            cv = cv + qdry * CVdry
            d = pt * ( dens * r * pt / P0 )**(r/cv)
            d = np.log(d)
        data[:,nv,:,:,:] = ( d - mean[var] ) / stddev[var]
        #print(var,data[:,nv,:,:,:].mean(),data[:,nv,:,:,:].std())
    file.close()
    for n in range(nt):
        t = hour + 9.0 + (it0 + n) * 30.0/3600.0
        th = (t / 24.0) * math.pi * 2.0
        #print(hour, n, t, th)
        data[n,nvar  ,:,:,:] = math.cos(th)
        data[n,nvar+1,:,:,:] = math.sin(th)
    return torch.from_numpy(data)




n0 = ndata_ts * myrank
fid = n0 // nt1
it0 = n0 % nt1
it1 = it0 + ndata_ts
if it1 > nt1:
    print(f"rank size is invalid {n0} {fid} {it0} {it1} {ndata_ts}")
    exit()

if myrank==0:
    print("read training data", flush=True)
    start = time.time()

data_t = get_var(fname_t[fid], hour_t[fid], it0, it1+1)

if myrank==0:
    print("shape of data_t", data_t.size(), flush=True)

if myrank==0:
    print(f"end (%.2f s)"%(time.time() - start), flush=True)

#exit()




ir = rank_size // nexp
iw = math.ceil(nt1 / ir)
fid = myrank // ir
i0 = min(iw * (myrank % ir), nt1)
i1 = min(i0 + iw, nt1)
ndata_es = i1 - i0

if myrank==0:
    print("read evaluation data", flush=True)
    start = time.time()

if ndata_es > 0:
    data_e = get_var(fname_e[fid], hour_e[fid], i0, i1+1)

if myrank==0:
    print(f"end (%.2f s)"%(time.time() - start), flush=True)



dx = 500.0
net = net2.Net2(vars, nvar, 2, dx, dz, nbuf, qmin, mean, stddev, stddev2)


net.criterion = nn.MSELoss()

#lr = 0.000001 * batch_size * rank_size
lr = 1e-4
#lr = 3e-5
#lr = 1e-5
#lr = 3e-6
#lr = 1e-6
optimizer = optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.Adam(net.parameters(), lr=0.01)
#optimizer = optim.Adam(net.parameters(), lr=0.001)


min_lt = 999.9e10
#min_lt = 0.708521

if pth:
    stat = torch.load(pth)
    net.load_state_dict(stat['net'])
    optimizer.load_state_dict(stat['opt'])
    min_lt = stat['min_lt']

#min_lt -= 0.05

for g in optimizer.param_groups:
    g['lr'] = lr

optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999)
#if pth:
#    scheduler.load_state_dict(stat['sch'])


if myrank==0:
    print("start training", flush=True)

start = time.time()

loss_t = np.zeros(max_epoch//nint + 1)
loss_e = np.zeros(max_epoch//nint + 1)
#ofact_t = np.zeros(max_epoch//nint + 1)




if myrank==0:
    #ofname = f"training_1step_nitr{nitr}_bsize{batch_size}"
    ofname = f"training_1step"
    if step:
        step = int(step)
        path = f"pth_1step/{step:02d}"
        os.makedirs(path)
    else:
        path = "."
    ofname = f"{path}/{ofname}"
    print(ofname)

def save(state, fname):
#    np.savez(fname, loss_t, loss_e, ofact_t)
    np.savez(fname, loss_t, loss_e)
    torch.save(state, fname + ".pth")



min_tmp = 999.9e10
min_dat = [min_tmp, 0, 0]

unchange = 0

ofact = -999.9

state = None
for epoch in range(max_epoch):
    if debug and myrank==0:
        print(f"epoch {epoch+1} / {max_epoch}", flush=True)

    net.train()
    net.drop = True
#    net.drop = False

#    if min_lt > 1.0:
#        ofact = 0.0
#    else:
#        ofact = min( ( 1.0 - min_lt ) * 1.1, 1.0 )

    optimizer.zero_grad()
    running_loss_t = 0.0
    idxs = list(range(ndata_ts))
    random.shuffle(idxs)
    for i in range(batch_num):
        if debug and myrank==0:
            print(f"batch {i+1} / {batch_num}", flush=True)
        i0 = batch_ts * i
        i1 = batch_ts * (i+1)
        idx = idxs[i0:i1]
        d_in = data_t[idx,:,:,:,:]
        out = net(d_in)
        loss = 0.0
        idxn = [l+1 for l in idx]
        #print(i, idx, idxn)
        for nv in range(nvar):
            var = vars[nv]
            obs = data_t[idxn,nv,:,:,:]
            tend = obs - d_in[:,nv,:,:,:]
            dat = out[:,nv,:,:,:]
            loss += net.get_loss(obs, tend, dat, var, ofact)

        del out
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
        #if debug:
        #    print(myrank, loss.item(), flush=True)
        running_loss_t += loss.item()
        #net.sync_norm()

    loss = torch.tensor(running_loss_t / batch_num)
    running_loss_t = hvd.allreduce(loss)

    #scheduler.step()

    if (epoch+1)%nint == 0 or epoch==0:

        net.eval()
        net.drop = False
        #print("sync", myrank)
        #net.sync_norm()

        if ndata_es > 0:
            with torch.no_grad():
                d_in = data_e[:-1,:,:,:,:]
                out = net(d_in)
                loss = 0.0
                for nv in range(nvar):
                    var = vars[nv]
                    obs = data_e[1:,nv,:,:,:]
                    tend = obs - d_in[:,nv,:,:,:]
                    dat = out[:,nv,:,:,:]
                    loss += net.get_loss(obs, tend, dat, var, ofact)
                loss = loss * ndata_es
        else:
            loss = torch.tensor(0.0)

        #print(f"eval rank {myrank}: loss {loss.item()}")
        #loss = hvd.allreduce(loss, average=False)
        loss = hvd.allreduce(loss, op=hvd.mpi_ops.Sum)

        #if myrank==0:
            #print(f"eval total loss {loss.item()}")
        running_loss_e = loss.item() / ndata_e

        l_e = running_loss_e
        l_t = running_loss_t

        if myrank==0:
            loss_t[(epoch+1)//nint] = l_t
            loss_e[(epoch+1)//nint] = l_e
#            ofact_t[(epoch+1)//nint] = ofact

        if epoch > 0 and ( l_e < min_dat[0] or l_t < min_lt ):
            min_dat = [l_e, l_t, epoch+1]
            unchange = 0
            if myrank==0:
                state = {
                    'net': net.state_dict(),
                    'opt': optimizer.state_dict(),
                    'sch': scheduler.state_dict(),
                    'min_lt': min_lt
                }

        if l_t < min_lt:
            min_lt = l_t

        if l_e < min_tmp:
            min_tmp = l_e

        if (epoch+1)%(math.ceil(max_epoch/neval)) == 0 or epoch==0:
            if myrank==0:
                for g in optimizer.param_groups:
                    lr = g['lr']
                #lr = scheduler.get_last_lr()[0]
                print('[%d] lr: %.2e, training: %.6f, eval: %.6f (%.6f, %.6f) (%.6f, %.6f)' % (epoch + 1, lr, l_t, l_e, min_tmp, min_dat[0], min_lt, ofact), flush=True)
                if state is not None:
                    save(state, ofname)
                    state = None
            if min_tmp > min_dat[0]:
                unchange += 1
#            if min_tmp > min_dat[0] * 1.5 or unchange >= 5:
#            if unchange >= 5:
#                break
            min_tmp = 999.9e10




#print(min_dat)
if myrank==0:
    print("minimam loss: %.6f, %.6f, %.6f, %d"%(min_dat[0], min_dat[1], ofact, min_dat[2]))
    print(f"elapsed time: %d s"%(time.time() - start))
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'sch': scheduler.state_dict(),
        'min_lt': min_lt
    }
    save(state, ofname+"_fin")
