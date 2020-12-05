import os
import torch
import numpy as np
import os.path as osp

from networks.d2f import D2F
from networks.loss_gl3d import criterion
from data.dl_gl3d import GL3D
from options.opt_gl3d import Options

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# Config
opt = Options().parse()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
cuda = torch.cuda.is_available()
opt.device = "cuda:0" if cuda else "cpu"

# Load the dataset
train_set = GL3D(data_path=opt.data_dir, train=True)
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)

# Model and Optimizer
d2net = D2F(opt.norm_mode)
d2net.to(opt.device)

param_list = list(d2net.parameters())
optimizer = torch.optim.SGD(param_list, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1 ** (1 / 200000))

# Train
d2net.train()
acc_list = []
loss_list = []
total_steps = 0
writer = SummaryWriter(log_dir=opt.runs_dir)
for epoch in range(opt.epochs):
    for _, train_data in enumerate(train_loader):
        total_steps += 1
        for idx, items in enumerate(train_data):
            if type(items) == list:
                train_data[idx] = [item.to(opt.device) for item in items]
            else:
                train_data[idx] = items.to(opt.device)

        with torch.set_grad_enabled(True):
            output = d2net(torch.cat([train_data[0], train_data[1]], dim=0), opt)
            train_loss, train_acc = criterion(output, train_data[2:], opt)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_list.append(train_acc)
            loss_list.append(train_loss.item())

        if total_steps == 1 or total_steps % opt.print_freq == 0:
            mean_acc = np.mean(acc_list)
            mean_loss = np.mean(loss_list)
            tmp_acc = []
            tmp_loss = []
            writer.add_scalar('train_loss', mean_loss, total_steps)
            writer.add_scalar('train_acc', mean_acc, total_steps)
            print('Train: total_steps {:d}, val_loss {:f}, train_acc {:f}'.format(total_steps, mean_loss, mean_acc))

        if total_steps == 1 or total_steps % opt.save_freq == 0:
            filename = osp.join(opt.models_dir, 'ckpt_{:07d}.pth.tar'.format(total_steps))
            checkpoint_dict = {'step': total_steps,
                               'd2net_state_dict': d2net.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint_dict, filename)
        scheduler.step()
writer.close()

