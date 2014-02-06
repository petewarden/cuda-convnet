# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy.random as nr
import numpy as n
import random as r
import sys
from time import time

IMAGE_SIZE_RAW=256
IMAGE_SIZE_TEST=224

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = IMAGE_SIZE_CIFAR
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        
        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

class CroppedRawDataProvider(LabeledRawDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledRawDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = IMAGE_SIZE_RAW - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        
        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,IMAGE_SIZE_RAW,IMAGE_SIZE_RAW))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        start_time = time()
        epoch, batchnum, datadic = LabeledRawDataProvider.get_next_batch(self)
        #sys.stderr.write('File loading took %s secs\n' % (time() - start_time))

        start_time = time()

        datadic['data'] = n.require(datadic['data'], requirements='C')
        datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1, datadic['data'].shape[1])), (1, self.data_mult)), requirements='C')

        cropped = n.zeros((self.get_data_dims(), datadic['data'].shape[1]*self.data_mult), dtype=n.single)

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1

        #sys.stderr.write('Other get_next_batch() work took %s secs\n' % (time() - start_time))

        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, IMAGE_SIZE_RAW, IMAGE_SIZE_RAW, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
            endY, endX = startY + self.inner_size, startX + self.inner_size
            pic = y[:,startY:endY,startX:endX, :]
            if nr.randint(2) == 0: # also flip the image with 50% probability
                pic = pic[:, :, ::-1, :]
            target = pic.copy().reshape((self.get_data_dims(), x.shape[1]))

class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1

class TestDataProvider(LabeledRawDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        self.curr_epoch = init_epoch
        if init_batchnum is None:
          init_batchnum = 0
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
        self.batch_range = 1
        self.inner_size = IMAGE_SIZE_TEST
        self.test_pattern = dp_params['test_pattern']
        self.label_count = dp_params['label_count']
        if self.test_pattern == 'solid':
          all_label_names = ['Black', 'Red', 'Green', 'Yellow', 'Blue', 'Purple', 'Cyan', 'White']
        elif self.test_pattern == 'stripes':
          all_label_names = ['Vertical', 'Horizontal']
        else:
          raise OptionException('TestDataProvider: Unknown test-pattern %s\n' % (self.test_pattern))
        if self.label_count > len(all_label_names):
          raise OptionException('TestDataProvider: Too many requested labels %d vs max %d for %s\n' % (self.label_count, len(all_label_names), self.test_pattern))
        my_label_names = all_label_names[0:self.label_count]
        self.batch_meta = {'label_names': my_label_names}

    def get_next_batch(self):
        num_cases = self.label_count
        images_data = n.empty((self.get_data_dims(), num_cases), dtype=n.float32)
        if self.test_pattern == 'solid':
          for i in range(num_cases):
            if i & 1 == 1:
              red = 255.0
            else:
              red = 0.0
            if i & 2 == 2:
              green = 255.0
            else:
              green = 0.0
            if i & 4 == 4:
              blue = 255.0
            else:
              blue = 0.0
            channel_stride = (IMAGE_SIZE_TEST * IMAGE_SIZE_TEST)
            images_data[0:channel_stride, i] = red * n.ones((self.get_data_dims() / 3), dtype=n.float32)
            images_data[channel_stride:(channel_stride*2), i] = green * n.ones((self.get_data_dims() / 3), dtype=n.float32)
            images_data[(channel_stride*2):(channel_stride*3), i] = blue * n.ones((self.get_data_dims() / 3), dtype=n.float32)
        elif self.test_pattern == 'stripes':
          for i in range(num_cases):
            square_view = images_data[:, i].reshape(IMAGE_SIZE_TEST, IMAGE_SIZE_TEST, 3)
            if i & 1 == 1:
              square_view[0::2, :, :] = 255.0 * n.ones(((IMAGE_SIZE_TEST * IMAGE_SIZE_TEST * 3) / 2), dtype=n.float32)
              square_view[1::2, :, :] = 0.0 * n.ones(((IMAGE_SIZE_TEST * IMAGE_SIZE_TEST * 3) / 2), dtype=n.float32)
            else:
              square_view[:, 0::2, :] = 255.0 * n.ones(((IMAGE_SIZE_TEST * IMAGE_SIZE_TEST * 3) / 2), dtype=n.float32)
              square_view[:, 1::2, :] = 0.0 * n.ones(((IMAGE_SIZE_TEST * IMAGE_SIZE_TEST * 3) / 2), dtype=n.float32)
        else:
          raise OptionException('TestDataProvider: Unknown test-pattern %s\n' % (self.test_pattern))
        labels = n.empty((1, num_cases), dtype=n.float32)
        for i in range(num_cases):
          labels[0, i] = i
        epoch = self.curr_epoch
        batchnum = self.curr_batchnum
        self.advance_batch()
        return epoch, batchnum, [images_data, labels]

    def advance_batch(self):
        self.curr_batchnum += 1
        if self.curr_batchnum >= self.batch_range:
          self.curr_batchnum = 0
          self.curr_epoch += 1

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class LabelSubsetProvider(CroppedRawDataProvider):

    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        CroppedRawDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.label_count = dp_params['label_count']
        original_label_names = self.batch_meta['label_names']
        self.batch_meta['label_names'] = original_label_names[0:self.label_count]

    def get_next_batch(self):
        how_many_loops = 0
        while True:
            epoch, batchnum, data = CroppedRawDataProvider.get_next_batch(self)
            all_images, labels = data
            num_images = all_images.shape[1]
            wanted_image_count = 0
            for i in range(num_images):
                if labels[0, i] < self.label_count:
                    wanted_image_count += 1
            if wanted_image_count > 0:
                break
            how_many_loops += 1
            if how_many_loops > 100:
                sys.stderr.write('Got into an infinite loop in LabelSubsetProvider.get_next_batch(), with a label_count of %d\n' % (self.label_count))
                exit(1)
        subset_images = n.empty((all_images.shape[0], wanted_image_count), dtype=n.float32)
        subset_labels = n.empty((1, wanted_image_count), dtype=n.float32)
        subset_index = 0
        for i in range(num_images):
          if labels[0, i] < self.label_count:
            subset_images[:, subset_index] = all_images[:, i]
            subset_labels[0, subset_index] = labels[0, i]
            subset_index += 1
        return epoch, batchnum, [subset_images, subset_labels]
