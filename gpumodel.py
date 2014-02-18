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
import os
from time import time, asctime, localtime, strftime
from numpy.random import randn, rand
import numpy.random as nr
from numpy import s_, dot, tile, zeros, ones, zeros_like, array, ones_like
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
from os import linesep as NL
import layer as lay
import binary
import numpy as n

class ModelStateException(Exception):
    pass

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=None, dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.dp_params = dp_params
        self.get_gpus()
        self.fill_excused_options()
        #assert self.op.all_values_given()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)

        # these are things that the model must remember but they're not input parameters
        if load_dic:
            self.model_state = load_dic["model_state"]
            self.save_file = self.options["load_file"].value
            if not os.path.isdir(self.save_file):
                self.save_file = os.path.dirname(self.save_file)
        else:
            self.model_state = {}
            if filename_options is not None:
                self.save_file = model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]

        self.init_data_providers()
        if load_dic: 
            self.train_data_provider.advance_batch()
            
        # model state often requries knowledge of data provider, so it's initialized after
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)
            
        self.import_model()
        self.init_model_lib()
        if self.print_entire_array:
            self.print_layer_weights(self.print_entire_array)
        
    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name) 
                   
    def fill_excused_options(self):
        pass
    
    def init_data_providers(self):
        self.dp_params['convnet'] = self
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batchnum"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()
        
    def init_model_state(self):
        pass
       
    def init_model_lib(self):
        pass
    
    def start(self):
        if self.test_only:
            #self.test_outputs += [self.get_test_error()]
            #self.print_test_results()
            self.print_predictions()
            sys.exit(0)
        self.train()
    
    def train(self):
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
        print "========================="
        next_data = self.get_next_batch()
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.image_count = data[2][0].shape[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            self.print_train_results()
            self.save_filter_image()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.get_test_error()]
                self.print_test_results(print_entire_array=self.print_entire_array)
                self.print_test_status()
                self.conditional_save()
            
            self.print_train_time(time() - compute_time_py)
        self.cleanup()
    
    def cleanup(self):
        sys.exit(0)
        
    def sync_with_host(self):
        self.libmodel.syncWithHost()
            
    def print_model_state(self):
        pass
    
    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1
    
    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=train)
    
    def parse_batch_data(self, batch_data, train=True):
        return batch_data[0], batch_data[1], batch_data[2]['data']
    
    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[2], not train)
    
    def finish_batch(self):
        return self.libmodel.finishBatch()
    
    def print_iteration(self):
        print "\t%d.%d... (%d images)" % (self.epoch, self.batchnum, self.image_count),
    
    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
    
    def print_train_results(self):
        batch_error = self.train_outputs[-1][0]
        if not (batch_error > 0 and batch_error < 2e20):
            print "Crazy train error: %.6f" % batch_error
            self.cleanup()

        print "Train error: %.6f " % (batch_error),

    def print_test_results(self):
        batch_error = self.test_outputs[-1][0]
        print "%s\t\tTest error: %.6f" % (NL, batch_error),

    def print_test_status(self):
        status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
        print status,
        
    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,
    
    def aggregate_test_outputs(self, test_outputs):
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if self.test_one else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error
    
    def get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            self.start_batch(data, train=False)
            load_next = not self.test_one and data[1] < self.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1]))
                print "%s" % (str(test_outputs))
            if not load_next:
                break
            sys.stdout.flush()
            
        return self.aggregate_test_outputs(test_outputs)

    def print_predictions(self):
        data = self.get_next_batch(train=False)[2] # get a test batch
        num_classes = self.test_data_provider.get_num_classes()
        softmax_idx = self.get_layer_idx('probs', check_type='softmax')
        NUM_IMGS = 1
        NUM_TOP_CLASSES = min(num_classes, 4) # show this many top labels
        label_names = self.test_data_provider.batch_meta['label_names']
        preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
        rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
        data[0] = n.require(data[0][:,rand_idx], requirements='C')
        data[1] = n.require(data[1][:,rand_idx], requirements='C')
        data += [preds]

        # Run the model
        self.libmodel.startFeatureWriter(data, softmax_idx)
        self.finish_batch()

        data[0] = self.test_data_provider.get_plottable_data(data[0])
        img_idx = 0
        true_label = int(data[1][0,img_idx])

        img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
        print "true_label=%s" % (label_names[true_label])
        for l in img_labels:
          print "l=%s" % (str(l))

        binary_checkpoint_file = "binary_%d.%d.ntwk" % (self.epoch, self.batchnum)
        binary_checkpoint_file_full_path = os.path.join(checkpoint_dir, binary_checkpoint_file)
        self.save_as_binary(binary_checkpoint_file_full_path)

    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val
        
    def get_var(self, var_name):
        return self.model_state[var_name]
        
    def has_var(self, var_name):
        return var_name in self.model_state
        
    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)
        
        dic = {"model_state": self.model_state,
               "op": self.op}
            
        checkpoint_dir = os.path.join(self.save_path, self.save_file)
        checkpoint_file = "%d.%d" % (self.epoch, self.batchnum)
        checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        pickle(checkpoint_file_full_path, dic,compress=self.zip_save)

        if self.do_binary_save:
          binary_checkpoint_file = "binary_%d.%d.ntwk" % (self.epoch, self.batchnum)
          binary_checkpoint_file_full_path = os.path.join(checkpoint_dir, binary_checkpoint_file)
          self.save_as_binary(binary_checkpoint_file_full_path)

        #for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
        #    if sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb*1024*1024 and f != checkpoint_file:
        #        os.remove(os.path.join(checkpoint_dir, f))
        #    else:
        #        break

    def save_as_binary(self, filename):
        layers = bytearray()
        for layer_dict in self.layers:
            type = layer_dict['type']
            if type not in lay.layer_savers:
              sys.stderr.write('save_as_binary(): Type not recognized: %s\n' % (type))
              continue
            sys.stderr.write('save_as_binary(): Type: %s\n' % (type))
            layer_saver = lay.layer_savers[type]
            layer_payload = layer_saver(layer_dict)
            layers.extend(layer_payload)
        graph = bytearray()
        graph.extend(binary.to_string('layers'))
        graph.extend(binary.to_list(layers))

        data_mean = self.train_data_provider.batch_meta['data_mean'].astype(n.float32)
        graph.extend(binary.to_string('data_mean'))
        graph.extend(binary.numpy_array_to_binary(data_mean))

        labels_payload = bytearray()
        label_names = self.train_data_provider.batch_meta['label_names']
        for label_name in label_names:
          labels_payload.extend(binary.to_string(label_name))
        graph.extend(binary.to_string('label_names'))
        graph.extend(binary.to_list(labels_payload))

        output = binary.to_dict(graph)
        file = open(filename, 'wb')
        file.write(output)
        file.close()


    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        op.add_option("save-path", "save_path", StringOptionParser, "Save path")
        op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[-1] * num_gpus"))
        op.add_option("print-entire-array", "print_entire_array", BooleanOptionParser, "Print all weight and bias values?", default=0)
        op.add_option("binary-save", "do_binary_save", BooleanOptionParser, "Save the network to a portable binary file in the checkpoints folder?", default=0)
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)
            
    def get_gpus(self):
        self.device_ids = [get_gpu_lock(g) for g in self.op.get_value('gpu')]
        if GPU_LOCK_NO_LOCK in self.device_ids:
            print "Not enough free GPUs!"
            sys.exit()
        
    @staticmethod
    def parse_options(op):
        try:
            load_dic = None
            options = op.parse()
            if options["load_file"].value_given:
                load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit()
        
