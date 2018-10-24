---
layout: post
title:  "DL Workflow Part2 - Local Experiments"
date:   2018-10-21 09:35:23 -0700
categories: dl_workflow
published: true
---

# Goals
- Create CIFAR 100 image classification dataset
  - Data is exported to mxnet recordio
- NVIDIA DALI data pipeline loading of recordio
- Training CIFAR 100 with Tensorboard output under various frameworks
    - mxnet - Resnet50_v2
    - keras - tensorflow - simple custom CNN
- Confirming that the libraries installed in Part 1 are functional

# Create /mnt/datasets

{% highlight bash %}
sudo groupadd datasets
sudo usermod -a -G datasets $USER
sudo mkdir -p /mnt/datasets
sudo chgrp -R datasets /mnt/datasets/
sudo chmod -R g+w /mnt/datasets/
exit
{% endhighlight %}

# CIFAR 100 Dataset

- 60000 32x32 colour images in 100 classes, with 600 images per class. 
- 50000 training
- 10000 test images. 
- 238MB extracted png images

<a href='{{ base }}/assets/source/deep_learning/cifar100_record.py'>Download cifar100_record.py</a>

{% highlight bash %}
mkdir -p /mnt/datasets/cifar100
cd /mnt/datasets/cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
python cifar100_record.py
python3 /usr/local/lib/python3.5/dist-packages/mxnet/tools/im2rec.py --no-shuffle --quality 9 --encoding .png --pack-label cifar_mxnet_train.lst  ./
python3 /usr/local/lib/python3.5/dist-packages/mxnet/tools/im2rec.py --no-shuffle --quality 9 --encoding .png --pack-label cifar_mxnet_test.lst   ./
{% endhighlight %}

Recordio database sizes

| File                  | ~Size Kb  | Note                         |
| --------------------- |----------| ---------------------------- |
| cifar_mxnet_test.idx  | 128      | Index file for test set      |
| cifar_mxnet_test.lst  | 799      | Test Set File list           |
| cifar_mxnet_test.rec  | 22000    | RecordIO Database            |
| cifar_mxnet_train.idx  | 714      | Index file for train set     |
| cifar_mxnet_train.lst  | 4100     | Train Set File list          |
| cifar_mxnet_train.rec  | 109000   | RecordIO Database            |

# Training CIFAR 100 using mxnet

<a href='{{ base }}/assets/source/deep_learning/cifar100_mxnet.py'>Download cifar100_mxnet.py</a>

{% highlight bash %}
rm -fr logs; rm -fr params; rm -f scalar_dict.json; python3 cifar100_mxnet.py --batch-size=64 --num-gpus 8 
sudo tensorboard --logdir=./logs --host=XXX.XXX.XXX.XXX --port=8888
{% endhighlight %}

**Note**: set num-gpus, batch-size, and ip arguments to match your server

## Tensorboard via mxboard

You can observe the progress of the training runs in tensorboard.

The goal is not to create an expressive model but to confirm that the system is functional.

![mxnet Training Progress](/assets/images/cifar100_mxnet_tensorboard_training.jpg){:class="img-responsive"}
![mxnet Images](/assets/images/cifar100_mxnet_tensorboard_images.jpg){:class="img-responsive"}

# Training CIFAR 100 using keras and tensorflow

<a href='{{ base }}/assets/source/deep_learning/cifar100_keras_tf.py'>Download cifar100_keras_tf.py</a>

{% highlight bash %}
rm -fr logs; python3 cifar100_keras_tf.py --batch-size=64 
sudo tensorboard --logdir=./logs --host=XXX.XXX.XXX.XXX --port=8888
{% endhighlight %}

**Note**: set batch-size and ip to match your server configuration

## Tensorboard via keras callback

You can observe the progress of the training runs in tensorboard.

The goal is not to create an expressive model but to confirm that the system is functional.

![Keras Compute Graph](/assets/images/cifar100_keras_graph.jpg){:class="img-responsive"}
![Keras Training Progress](/assets/images/cifar100_keras_training.jpg){:class="img-responsive"}

# Next Steps

The system installation and some basic deep learning is complete. 
A good idea at this point would be to backup the system to lock in all of the libraries and dependencies. 

Servers at cirrascale can be configured in any redundant storage configuration you desire.
Moreover, we can provide access to remote storage and provision your system with it mounted.

We are happy to help you craft a backup solution that meets your specific requirements.

[Contact Us Today](mailto:info@cirrascale.com)