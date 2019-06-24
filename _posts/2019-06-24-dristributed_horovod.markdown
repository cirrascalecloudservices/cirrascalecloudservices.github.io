---
layout: post
title:  "Distributed Horovod - NCCL Tuning in Bonded Enviornments"
date:   2019-06-24 09:30:00 -0700
categories: horovod, distributed_training
published: true
---

# Overview/Goal
- Train a tensorflow model distributed amongst many nodes with each containing many gpus
- Use [Horovod](https://github.com/horovod/horovod) as the distributed training framework
- Specify the interface that NCCL uses to send its broadcast/reduce traffic
  - Specify which vlan interface to use during with bonded physical interfaces

# Horovod
- Distributed Tensorflow execution using OpenMPI as coordinator
  - Tensorflow uses NCCL under the covers to determine most efficient pathing
- Easier to configure than manual configuratiuon of a distributed TensorFlow cluster

# Relevant [NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html) Tunables

- NCCL_SOCKET_IFNAME
  - The name of the network interface that NCCL will use for inter-node communication.
- NCCL_P2P_LEVEL
  - GPU peering configuration.
    - Defaults to 3 = Use P2P when GPUs are on the same PCI root complex

# Relevant [Open MPI](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php) Tunables

- btl_tcp_if_exclude
  - The name of the network interfaces that should be excluded from openmpi communications.

# Starting the Horovod Server [Uber's Horovod Docker Image]

{% highlight bash %}
nvidia-docker run 
--privileged 
--network=host 
--name main_server 
-v /Data/ImagenetData/Tf_Protobuf352:/imagenet_tfrecords
--rm -it uber/horovod:0.13.10-tf1.9.0-torch0.4.0-py2.7
bash -c "/usr/sbin/sshd -p 38798; sleep infinity"
{% endhighlight %}

# Starting the Horovod Worker [Uber's Horovod Docker Image]

{% highlight bash %}
nvidia-docker run 
--privileged 
--network=host 
--name main_worker 
--hostname main_worker 
-v /raid/ImagenetData/Tf_Protobuf352:/imagenet_tfrecords
--rm -it uber/horovod:0.13.10-tf1.9.0-torch0.4.0-py2.7
{% endhighlight %}

# Horovod Server - Naive Launch of MPI Job from inside the container

Starting the horovod job using a default set of arguments:

{% highlight bash %}
mpirun
-np 9
-H localhost:8,172.18.0.100:1
-mca plm_rsh_args "-p 38798"
-bind-to none
-map-by slot
-mca pml ob1
-mca btl ^openib
-x NCCL_DEBUG=INFO
-x LD_LIBRARY_PATH
-x PATH python nvcnn_hvd.py
-m inception3
-b 64
--num_batches 500
--display_every 10
--fp16
{% endhighlight %}

**Note:** nvcnn_hvd.py is from tensorflow developers and is the "hello world" that ships with the nvidia cloud containers.

**Training Network Traffic**

![Default Horovod Bandwidth](/assets/images/horovod_default_launch_throughput.jpg){:class="img-responsive"}

- bond0 - physical bond of a dual port connectx-4
- bond0.AAAA - vlan that carries public internet traffic
- bond0.BBBB - vlan that is dedicated to the cluster

**Observations:**
- The traffic is symmetic on the bond0.AAAA and bond0.BBBB
  - Expectation is that NCCL would have selected bond0.BBBB for all its traffic.
  - bond0.AAAA is lexicographically before bond0.BBBB
    - Both bonded vlans are on the same physical interface 
    - It seems likely that the nccl would select bond0.AAAA as the default interface if using a sorted list by name

**Potential Optimizations / Things to Try**
- Set the interfaces that MPI uses for inter node communications - btl_tcp_if_exclude
- Set the interface that NCCL uses for inter node communications - NCCL_SOCKET_IFNAME

# Horovod Server - Specific Launch Specifying NCLL and OpenMPI Interfaces

{% highlight bash %}
mpirun
-np 9
-H localhost:8,172.18.0.100:1
--mca btl_tcp_if_exclude docker0,lo,bond0.AAAA
-mca plm_rsh_args "-p 38798"
-bind-to none
-map-by slot
-mca pml ob1
-mca btl ^openib
-x NCCL_SOCKET_IFNAME=bond0.BBBB
-x NCCL_DEBUG=INFO
-x LD_LIBRARY_PATH
-x PATH python nvcnn_hvd.py
-m inception3
-b 64
--num_batches 500
--display_every 10
--fp16
{% endhighlight %}

The openmpi argument mca btl_tcp_if_exclude tells openmpi to exclude docker, localhost, and bond0.AAAA interfaces.

The NCCL environment variable NCCL_SOCKET_IFNAME tells NCCL which interface to use for internode communications.

**Training Network Traffic**

![Optimized Horovod Bandwidth](/assets/images/horovod_optimized_launch_throughput.jpg){:class="img-responsive"}

**Observations:**
- Traffic on bond0.AAAA is greatly reduced
- Traffic on bond0.BBBB is higher than when using the defaults
  - Training speed is also slightly ~5% increased and also less jittery.

# Conclusion

- Depending on your cluster's networking enviornment manual configuration of NCCL and Open MPI parameters may be warranted.
- NCCL's default "best interface" selection method may send out traffic on an interface you were not expecting.
  - Monitor your network training traffic using [iptraf-ng](https://packages.ubuntu.com/xenial/iptraf-ng).