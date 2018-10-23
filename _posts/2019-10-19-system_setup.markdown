---
layout: post
title:  "DL Workflow Part1 - Environment Setup"
date:   2018-10-19 09:35:23 -0700
categories: dl_workflow
published: true
---

# Goals
- Setup a Ubuntu 16.04 LTS Server for Deep Learning & Data Exploration
- System Smoke Check
- Using Cloud-init on Cirrascale Cloud Services

# Ubuntu 16.04 LTS Server Setup for Deep Learning

Prototypical Deep Learning Stack
- Python 3.5
- Python Data Science Libraries
- Python DL Frameworks
  - Tensorflow
  - Pytorch
  - Mxnet
  - Keras
- Nvidia-docker2
- R 
- R Data Science Libraries
  - tidyverse
  - shiny
  - threejs
- R DL Frameworks
  - Tensorflow
  - Keras

All follow up blogs require this base configuration.

Note: Ubuntu 18.04 does not offically support gcc-6 that is needed to install prebuilt TF python module.

Why Cuda 9.0?

It is the lowest common denominator that works with all the frameworks AND prebuilt packages.
Updating to Cuda 9.X or 10.X is possible but all libraries will need to be compiled from source [a task that is better left to another article.]

## system libraries and utilities
{% highlight bash %}
sudo apt-get update
sudo apt-get install -y build-essential dkms
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
sudo apt-get install -y git htop iftop iotop jq
sudo apt-get install -y libssl-dev libssh2-1-dev  libcurl4-gnutls-dev libgit2-dev libxml2-dev libsqlite0-dev sqlite
sudo apt-get install -y python3
sudo apt-get install -y python3-dev
sudo apt-get install -y libopenblas-base r-base
sudo reboot now
{% endhighlight %}

## cuda 9.0 + cudnn 7.0
{% highlight bash %}
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod +x cuda_9.0.176_384.81_linux-run
sudo ./cuda_9.0.176_384.81_linux-run --silent --driver --samples --toolkit
echo "/usr/local/cuda-9.0/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig

#cudnn requires an account and manual download from https://developer.nvidia.com
#cudnn-9.0-linux-x64-v7.solitairetheme8
tar -xvf cudnn-9.0-linux-x64-v7.solitairetheme8
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda-9.0/lib64/libcudnn*
{% endhighlight %}

## docker-ce repo
{% highlight bash %}
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
{% endhighlight %}

## nvidia-docker repo
{% highlight bash %}
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
{% endhighlight %}

## docker-ce and nvidia-docker
{% highlight bash %}                                                                                    
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo usermod -aG docker $USER
{% endhighlight %}

## python libraries
{% highlight bash %}
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

sudo pip3 install scikit-learn
sudo pip3 install scikit-image
sudo pip3 install opencv-python
sudo pip3 install matplotlib
sudo pip3 install pandas
sudo pip3 install mxnet-cu90
sudo pip3 install mxnet
sudo pip3 install mxboard
sudo pip3 install keras
sudo pip3 install torchvision
#if you have an nvidia gpu
sudo pip3 install tensorflow-gpu
sudo pip3 install tensorflow
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali
#if running virtualized without AVX
#sudo pip3 install tensorflow==1.5
{% endhighlight %}

## R libraries
{% highlight bash %}
echo "install.packages(\"devtools\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"tidyverse\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"data.table\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"dplyr\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"RSQLite\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"shiny\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"htmlwidgets\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"leaflet\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "install.packages(\"threejs\", repos=\"https://cran.rstudio.com\")" | sudo R --no-save
echo "devtools::install_github(\"rstudio/keras\")" | sudo R --no-save
{% endhighlight %}

System setup is now complete. Reboot and run smoke tests.

{% highlight bash %}
sudo updatedb
sudo reboot now
{% endhighlight %}

This is the baseline configuration that will be used for articles in this blog series.

# System Smoke Tests

Let's kick the tires.

## nvidia-docker
{% highlight bash %}
docker run hello-world
docker image ls
{% endhighlight %}

## tensorflow smoketest
{% highlight bash %}
python3 -c 'from __future__ import print_function; import tensorflow as tf; hello = tf.constant("Hello!"); sess = tf.Session(); print (sess.run(hello))'
{% endhighlight %}

## pytorch smoketest
{% highlight bash %}
python3 -c 'from __future__ import print_function; import torch; x=torch.empty(1,1); print(x)'
{% endhighlight %}

## mxnet smoketest
{% highlight bash %}
python3 -c 'from __future__ import print_function; import mxnet as mx; mx.cpu()'
{% endhighlight %}

## R smoketest
{% highlight bash %}
echo "library(\"keras\")" | sudo R --no-save
{% endhighlight %}

Everything looks good. Time to light the fires!

# Using Cloud-init on Cirrascale Cloud Services

At Cirrascale we love baremetal and don't like repeating ourselves.

Cloud-init handles configuring systems for our users.

https://cloudinit.readthedocs.io/en/latest/ 