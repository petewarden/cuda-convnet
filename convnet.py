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

import numpy as n
import numpy.random as nr
from util import *
from data import *
from options import *
from gpumodel import *
import sys
import math as m
import layer as lay
from convdata import *
from os import linesep as NL

import matplotlib
matplotlib.use('Agg')

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
    sys.exit(1)

class ConvNet(IGPUModel):
    def __init__(self, op, load_dic, dp_params={}):
        filename_options = []
        self.animation_image_index = 0
        dp_params['multiview_test'] = op.get_value('multiview_test')
        dp_params['crop_border'] = op.get_value('crop_border')
        dp_params['label_count'] = op.get_value('label_count')
        dp_params['test_pattern'] = op.get_value('test_pattern')
        dp_params['image_size'] = op.get_value('image_size')
        IGPUModel.__init__(self, "ConvNet", op, load_dic, filename_options, dp_params=dp_params)
        
    def import_model(self):
        lib_name = "pyconvnet" if is_windows_machine() else "_ConvNet"
        print "========================="
        print "Importing %s C++ module" % lib_name
        self.libmodel = __import__(lib_name) 
        
    def init_model_lib(self):
        self.libmodel.initModel(self.layers, self.minibatch_size, self.device_ids[0])
        
    def init_model_state(self):
        ms = self.model_state
        if self.load_file:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self, ms['layers'])
        else:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self)
        self.layers_dic = dict(zip([l['name'] for l in ms['layers']], ms['layers']))
        
        logreg_name = self.op.get_value('logreg_name')
        if logreg_name:
            self.logreg_idx = self.get_layer_idx(logreg_name, check_type='cost.logreg')
        
        # Convert convolutional layers to local
        if len(self.op.get_value('conv_to_local')) > 0:
            for i, layer in enumerate(ms['layers']):
                if layer['type'] == 'conv' and layer['name'] in self.op.get_value('conv_to_local'):
                    lay.LocalLayerParser.conv_to_local(ms['layers'], i)
        # Decouple weight matrices
        if len(self.op.get_value('unshare_weights')) > 0:
            for name_str in self.op.get_value('unshare_weights'):
                if name_str:
                    name = lay.WeightLayerParser.get_layer_name(name_str)
                    if name is not None:
                        name, idx = name[0], name[1]
                        if name not in self.layers_dic:
                            raise ModelStateException("Layer '%s' does not exist; unable to unshare" % name)
                        layer = self.layers_dic[name]
                        lay.WeightLayerParser.unshare_weights(layer, ms['layers'], matrix_idx=idx)
                    else:
                        raise ModelStateException("Invalid layer name '%s'; unable to unshare." % name_str)
        self.op.set_value('conv_to_local', [], parse=False)
        self.op.set_value('unshare_weights', [], parse=False)
    
    def get_layer_idx(self, layer_name, check_type=None):
        try:
            layer_idx = [l['name'] for l in self.model_state['layers']].index(layer_name)
            if check_type:
                layer_type = self.model_state['layers'][layer_idx]['type']
                if layer_type != check_type:
                    raise ModelStateException("Layer with name '%s' has type '%s'; should be '%s'." % (layer_name, layer_type, check_type))
            return layer_idx
        except ValueError:
            raise ModelStateException("Layer with name '%s' not defined." % layer_name)

    def fill_excused_options(self):
        if self.op.get_value('check_grads'):
            self.op.set_value('save_path', '')
            self.op.set_value('train_batch_range', '0')
            self.op.set_value('test_batch_range', '0')
            self.op.set_value('data_path', '')
            
    # Make sure the data provider returned data in proper format
    def parse_batch_data(self, batch_data, train=True):
        if max(d.dtype != n.single for d in batch_data[2]):
            raise DataProviderException("All matrices returned by data provider must consist of single-precision floats.")
        return batch_data

    def start_batch(self, batch_data, train=True):
        data = batch_data[2]
        if self.check_grads:
            self.libmodel.checkGradients(data)
        elif not train and self.multiview_test:
            self.libmodel.startMultiviewTest(data, self.train_data_provider.num_views, self.logreg_idx)
        else:
            self.libmodel.startBatch(data, not train)
        
    def print_iteration(self):
        print "%d.%d... (%d images)" % (self.epoch, self.batchnum, self.image_count),

    def save_filter_image(self):
      if not self.show_filters:
        return
      self.plot_filters()
      image_file_name = "%s_%05d.png" % (self.show_filters, self.animation_image_index)
      image_file_path = os.path.join(self.save_path, image_file_name)
      pl.savefig(image_file_path)
      self.animation_image_index += 1

    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
        
    def print_costs(self, cost_outputs, do_exit_on_nan=True):
        costs, num_cases = cost_outputs[0], cost_outputs[1]
        for errname in costs.keys():
            costs[errname] = [(v/num_cases) for v in costs[errname]]
            print "%s: " % errname,
            print ", ".join("%6f" % v for v in costs[errname]),
            if sum(m.isnan(v) for v in costs[errname]) > 0 or sum(m.isinf(v) for v in costs[errname]):
                print "^ got nan or inf!"
                #if do_exit_on_nan:
                #  sys.exit(1)

    def print_train_results(self):
        self.print_costs(self.train_outputs[-1])
        
    def print_test_status(self):
        pass
        
    def print_test_results(self, print_entire_array=False):
        print ""
        print "======================Test output======================"
        self.print_costs(self.test_outputs[-1], do_exit_on_nan = False)
        print ""
        print "-------------------------------------------------------",
        self.print_layer_weights(print_entire_array)

    def print_layer_weights(self, print_entire_array = False):
        for i,l in enumerate(self.layers): # This is kind of hacky but will do for now.
            if 'weights' in l:
                if type(l['weights']) == n.ndarray:
                    print "%sLayer '%s' weights: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['weights'])), n.mean(n.abs(l['weightsInc']))),
                    if print_entire_array:
                        n.set_printoptions(threshold=100)
                        print "weights.shape=%s" % (str(l['weights'].shape))
                        print "weights=[%s]" % (str(['weights'])),
                        print "weightsInc=[%s]" % (str(l['weightsInc'])),
                elif type(l['weights']) == list:
                    print ""
                    print NL.join("Layer '%s' weights[%d]: %e [%e]" % (l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi))) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))),
                    if print_entire_array:
                      n.set_printoptions(threshold=100)
                      for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc'])):
                        print "weights.shape=%s" % (str(w.shape))
                        print "weights=[%s]" % (str(w)),
                        print "weightsInc=[%s]" % (str(wi)),
                print "%sLayer '%s' biases: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['biases'])), n.mean(n.abs(l['biasesInc']))),
                if print_entire_array:
                  n.set_printoptions(threshold=100)
                  print "biases.shape=%s" % (str(l['biases'].shape))
                  print "biases=[%s]" % (str(l['biases'])),
                  print "biasesInc=[%s]" % (str(l['biasesInc'])),
        print ""


    def conditional_save(self):
        self.save_state()
        print "-------------------------------------------------------"
        print "Saved checkpoint to %s" % os.path.join(self.save_path, self.save_file)
        print "=======================================================",
        
    def aggregate_test_outputs(self, test_outputs):
        num_cases = sum(t[1] for t in test_outputs)
        for i in xrange(1 ,len(test_outputs)):
            for k,v in test_outputs[i][0].items():
                for j in xrange(len(v)):
                    test_outputs[0][0][k][j] += test_outputs[i][0][k][j]
        return (test_outputs[0][0], num_cases)

    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
        FILTERS_PER_ROW = 16
        MAX_ROWS = 16
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
        else:
            bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
    
        for m in xrange(filter_start,filter_end ):
            filter = filters[:,:,m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c,:].reshape((filter_size,filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size,filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        filter_start = 0 # First filter to show
        layer_names = [l['name'] for l in self.layers]
        if self.show_filters not in layer_names:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[layer_names.index(self.show_filters)]
        filters = layer['weights'][self.input_idx]
        if layer['type'] == 'fc': # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], layer['filterPixels'][self.input_idx] * channels, num_filters))
                filter_start = r.randint(0, layer['modules']-1)*num_filters # pick out some random modules
                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
                num_filters *= layer['modules']

        filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0,:,:] + 1.28033 * filters[2,:,:]
            G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
            B = filters[0,:,:] + 2.12798 * filters[1,:,:]
            filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        filters = filters - filters.min()
        filters = filters / filters.max()

        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans)


    @classmethod
    def get_options_parser(cls):
        op = IGPUModel.get_options_parser()
        op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
        op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=True)
        op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
        op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path','save_path','train_batch_range','test_batch_range'])
        op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0, requires=['logreg_name'])
        op.add_option("crop-border", "crop_border", IntegerOptionParser, "Cropped DP: crop border size", default=4, set_once=True)
        op.add_option("logreg-name", "logreg_name", StringOptionParser, "Cropped DP: logreg layer name (for --multiview-test)", default="")
        op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
        op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
        op.add_option("conserve-mem", "conserve_mem", BooleanOptionParser, "Conserve GPU memory (slower)?", default=0)
        op.add_option("label-count", "label_count", IntegerOptionParser, "How many labels to choose in the subset case", default=2, set_once=True)
        op.add_option("test-pattern", "test_pattern", StringOptionParser, "What patterns to use for the synthesized tests", default="solid", set_once=True)
        op.add_option("image-size", "image_size", IntegerOptionParser, "The square size of the images", default=256, set_once=True)
        op.add_option("show-filters", "show_filters", StringOptionParser, "Save learned filters in specified layer to per-iteration image files", default="")
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)

        op.delete_option('max_test_err')
        op.options["max_filesize_mb"].default = 0
        op.options["testing_freq"].default = 50
        op.options["num_epochs"].default = 50000
        op.options['dp_type'].default = None
        
        DataProvider.register_data_provider('cifar', 'CIFAR', CIFARDataProvider)
        DataProvider.register_data_provider('dummy-cn-n', 'Dummy ConvNet', DummyConvNetDataProvider)
        DataProvider.register_data_provider('cifar-cropped', 'Cropped CIFAR', CroppedCIFARDataProvider)
        DataProvider.register_data_provider('raw-cropped', 'Cropped CIFAR', CroppedRawDataProvider)        
        DataProvider.register_data_provider('test', 'Test Data', TestDataProvider)
        DataProvider.register_data_provider('raw-subset', 'A limited set of the full raw labels', LabelSubsetProvider)

        return op
    
if __name__ == "__main__":
    #nr.seed(5)
    op = ConvNet.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = ConvNet(op, load_dic)
    model.start()
