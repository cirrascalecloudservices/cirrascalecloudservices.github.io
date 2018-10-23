import os
import numpy
import _pickle as cPickle
from optparse import OptionParser
from skimage.io import imsave

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def save_as_image(img_flat, fname):
    # consecutive 1024 pixel rows store color channels of 32x32 image
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = numpy.dstack((img_R, img_G, img_B))
    imsave(fname, img)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--directory", dest="dir", help="cifar10 directory", default="/mnt/datasets/cifar100/cifar-100-python")
    options, args = parser.parse_args()
    metadata=  unpickle( os.path.join(options.dir, "meta"))
    coarselabels = metadata['coarse_label_names']
    finelabels = metadata['fine_label_names']

    #loop over the test and train pickled dictionaries
    for dictname in ["test", "train"]:
        dictdata = unpickle( os.path.join(options.dir, dictname))
        datalen = len(dictdata['filenames'])
        print (dictname, datalen )
        mxnetlist = open('cifar_mxnet_{}.lst'.format(dictname),"w")
        for i in range (0,datalen,1) :
            coarseindex = dictdata['coarse_labels'][i]
            fineindex = dictdata['fine_labels'][i]
            coarsedir = coarselabels[dictdata['coarse_labels'][coarseindex]]
            finedir = finelabels[dictdata['fine_labels'][fineindex]]
            destdir = os.path.join(os.curdir, "output", dictname, coarsedir,finedir)
            destfile = os.path.join(destdir, dictdata['filenames'][i])
            imgdata = dictdata['data'][i]
            if not os.path.exists(destdir):
                os.makedirs(destdir)
            save_as_image(imgdata,destfile)
            mxnetlist.write('{} \t {} \t {}  \t {}\r\n'.format(i,coarseindex,fineindex,destfile))
        mxnetlist.close()
