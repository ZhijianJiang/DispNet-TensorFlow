#! /usr/bin/env python2
# wget -r -np -A png,pfm,pgm,txt http://vision.middlebury.edu/stereo/data/scenes2014/datasets/
# wget -r -np -A png,pfm,pgm,txt http://vision.middlebury.edu/stereo/data/scenes2006/FullSize/

import os
import re
import sys
import subprocess
import cPickle
import numpy as np

def load_pfm(fname, crop):
    if crop:
        if not os.path.isfile(fname + '.H.pfm'):
            x, scale = load_pfm(fname, False)
            x_ = np.zeros((384, 768), dtype=np.float32)
            for i in range(77, 461):
                for j in range(96, 864):
                    x_[i - 77, j - 96] = x[i, j]
            save_pfm(fname + '.H.pfm', x_, scale)
            return x_, scale
        else:
            fname += '.H.pfm'
    color = None
    width = None
    height = None
    scale = None
    endian = None
  
    file = open(fname)
    header = file.readline().rstrip()
    if header == 'PF':
        color = True    
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
 
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
 
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
 
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape)), scale

def save_pfm(fname, image, scale=1):
    file = open(fname, 'w') 
    color = None
 
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
 
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))
 
    endian = image.dtype.byteorder
   
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
 
    file.write('%f\n' % scale)
   
    np.flipud(image).tofile(file)

def tofile(fname, x):
    if x is None:
        open(fname + '.dim', 'w').write('0\n')
        open(fname, 'w')
    else:
        x.tofile(fname)
        open(fname + '.type', 'w').write(str(x.dtype))
        open(fname + '.dim', 'w').write('\n'.join(map(str, x.shape)))


output_dir = '/home/visg1/jzj/'
assert(os.path.isdir(output_dir))
dispnoc = []
base1 = '/home/visg1/jzj/Data_fin2/gt'

filenames = sorted(os.listdir(base1))

for i in range(np.size(filenames)):
	print(filenames[i])
	disp, scale = load_pfm(os.path.join(base1, filenames[i]), False)
	dispnoc.append(disp.astype(np.float32))


with open(os.path.join(output_dir, 'gopro2.pkl'), 'wb') as f:
	cPickle.dump(dispnoc, f)

#subprocess.check_call('rm -f {}/*.{{bin,dim,txt,type}}'.format(output_dir), shell=True)

# for i in range(len(dispnoc)):
#     tofile('{}/dispnoc_{:04d}.bin'.format(output_dir, i + 1), dispnoc[i])


#with open('./test.pkl', 'rb') as f:
	#aaa = cPickle.load(f)
