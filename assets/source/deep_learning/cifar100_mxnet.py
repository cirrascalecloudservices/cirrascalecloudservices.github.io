import matplotlib
matplotlib.use('Agg')

import argparse, time, logging
import numpy as np
import mxnet as mx
import cv2
import numpy as np

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
from gluoncv.utils import viz

import os
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.io import ImageRecordIter
from skimage.io import imsave

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
#https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html
parser.add_argument('--model', type=str, default='resnet50_v2',
                    help='model to use. options are resnet and wrn. default is resnet.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=64, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='period in epoch for learning rate decays. default is 0 (has no effect).')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                    help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--save-period', type=int, default=10,
                    help='period in epoch of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--resume-from', type=str,
                    help='resume training from the model')
parser.add_argument('--save-plot-dir', type=str, default='.',
                    help='the path to save the history plot')
opt = parser.parse_args()
logging.basicConfig(level=logging.ERROR)
logging.info(opt)

class DataIterLoader(object):
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label

    def next(self):
        return self.__next__() # for Python 2

batch_size = opt.batch_size
classes = 100

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

model_name = opt.model
if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes, 'drop_rate': opt.drop_rate}
else:
    kwargs = {'classes': classes}
net = get_model(model_name, **kwargs)

if opt.resume_from:
    net.load_parameters(opt.resume_from, ctx = context)
optimizer = 'nag'

save_period = opt.save_period
if opt.save_dir and save_period:
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
else:
    save_dir = ''
    save_period = 0

plot_path = opt.save_plot_dir

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.hybridize()

    train_data_iter = mx.io.ImageRecordIter(
      path_imgrec="cifar_mxnet_train.rec",
      data_shape=(3,32,32),
      path_imglist="cifar_mxnet_train.lst",
      batch_size=batch_size,
      shuffle= True
    )
    val_data_iter = mx.io.ImageRecordIter(
      path_imgrec="cifar_mxnet_test.rec",
      data_shape=(3,32,32),
      path_imglist="cifar_mxnet_test.lst",
      batch_size=batch_size,
      shuffle=False
    )
    train_data = DataIterLoader(train_data_iter)
    val_data = DataIterLoader(val_data_iter)

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    
    train_metric = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    #iteration = 0
    lr_decay_count = 0
    best_val_score = 0

    # collect parameter names for logging the gradients of parameters in each epoch
    params = net.collect_params()
    param_names = params.keys()

    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    global_step = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        num_batch = 10000
        #num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0],  ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

            loss_val = sum(loss[0]) / len (loss[0])
            loss_val = loss_val.asscalar()
            sw.add_scalar(tag='cross_entropy', value=loss_val, global_step=global_step)
            global_step += 1
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_metric.update(label, output)

            # Log the first batch of images of each epoch
            if i == 0:
                sw.add_image('cifar100_minibatch', data[0]/255, epoch)

        if epoch == 0:
            sw.add_graph(net)

        name, train_acc = train_metric.get()
        print('[Epoch %d] Training: %s=%f Time: %f' % (epoch, name, train_acc, time.time()-tic))
        # logging training accuracy
        sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc), global_step=epoch)

        name, val_acc = test(ctx,val_data)
        print('[Epoch %d] Validation: %s=%f Time: %f' % (epoch, name, val_acc, time.time()-tic))
        # logging the validation accuracy
        sw.add_scalar(tag='accuracy_curves', value=('valid_acc', val_acc), global_step=epoch)

        if val_acc > best_val_score:
            best_val_score = val_acc
            net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            net.save_parameters('%s/cifar100-%s-%d.params'%(save_dir, model_name, epoch))

    sw.export_scalars('scalar_dict.json')
    sw.close()

    if save_period and save_dir:
        net.save_parameters('%s/cifar100-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
