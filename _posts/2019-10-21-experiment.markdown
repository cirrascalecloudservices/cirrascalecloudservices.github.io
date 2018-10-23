---
layout: post
title:  "DL Workflow Part2 - Local Experiment"
date:   2018-10-21 09:35:23 -0700
categories: dl_workflow
published: false
---

# Goals
- Create CIFAR 10 image classification experiment
- 

# DataSets Folder

{% highlight bash %}
sudo groupadd datasets
sudo usermod -a -G datasets $USER
sudo mkdir -p /mnt/datasets
sudo chgrp -R datasets /mnt/datasets/
sudo chmod -R g+w /mnt/datasets/
exit
{% endhighlight %}

# CIFAR 10 Dataset

{% highlight bash %}
mkdir -p /mnt/datasets/cifar10
cd /mnt/datasets/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
{% endhighlight %}

# CIFAR 100 Dataset

{% highlight bash %}
mkdir -p /mnt/datasets/cifar100
cd /mnt/datasets/cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
python cifar100_record.py
python3 /usr/local/lib/python3.5/dist-packages/mxnet/tools/im2rec.py --pack-label --resize 256 cifar_mxnet_train.lst  ./
python3 /usr/local/lib/python3.5/dist-packages/mxnet/tools/im2rec.py --pack-label --resize 256 cifar_mxnet_test.lst   ./
{% endhighlight %}

# CIFAR 100 Train

Note: set num-gpus and batch-size arguments to match your server

{% highlight bash %}
python3 cifar100_mxnet.py --num-gpus 8 --batch-size=64
{% endhighlight %}