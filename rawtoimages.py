#!/usr/bin/python
#
# Copyright (c) 2014, Pete Warden
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

# This utility script takes a raw batch file, and converts it into a folder of images.
# If a batch.meta file is present in the same folder as the input raw file, then it will
# be read to figure out the label to include in each output image's filename.
#
# Usage is:
# python rawtoimages.py <raw batch file> <output folder> <image size>

import os
import sys
import glob
from scipy import misc
import re
import numpy as n
import pickle

def log(message):
  sys.stderr.write("%s\n" % (message))

log_counts = {}
def log_count(name, stride = 100):
  if name not in log_counts:
    log_counts[name] = 0
  log_counts[name] += 1
  if log_counts[name] % stride == 0:
    sys.stderr.write("%s %d\n" % (name, log_counts[name]))

if len(sys.argv) < 4:
  log('Usage: python rawtoimages.py <raw batch file> <output folder> <image size>\n')
  exit(1)

raw_name = sys.argv[1]
output_folder = sys.argv[2]
image_size = int(sys.argv[3])

if not os.path.exists(raw_name):
  log('raw batch file "%s" not found.\n' % (raw_name))
  exit(1)

input_folder = os.path.dirname(raw_name)
batches_meta_name = os.path.join(input_folder, 'batches.meta')

label_names = None
if os.path.exists(batches_meta_name):
  batches_meta_file = open(batches_meta_name, 'rb')
  batches_meta = pickle.load(batches_meta_file)
  batches_meta_file.close()
  label_names = batches_meta['label_names']

input_file = open(raw_name)
input_file.seek(0,2)
size = input_file.tell()
input_file.seek(0,0)
bytes_per_channel = (image_size * image_size)
bytes_per_image = (bytes_per_channel * 3)
bytes_per_image_plus_label = (bytes_per_image + 4)
image_count = (size / bytes_per_image_plus_label)
if (image_count * bytes_per_image_plus_label) != size:
  log('Bad file size %s for %s - expected %dx%dx%dx3 + %dx4\n' % (size, raw_name, image_count, image_size, image_size, image_count))
  exit(1)
mm = input_file.read(size)
input_file.close()
entry = {}
label_indexes = n.frombuffer(mm, dtype=n.float32, count = image_count)
all_images_data = n.frombuffer(mm,
  dtype=n.uint8,
  offset = (4 * image_count),
  count = (bytes_per_image * image_count)).reshape((bytes_per_image, image_count))

raw_base = os.path.basename(raw_name)

for image_index in range(image_count):
  label_index = int(label_indexes[image_index])
  if label_names:
    label_name = label_names[label_index]
  else:
    label_name = str(label_index)
  output_name = "%s_%04d_%s.png" % (raw_base, image_index, label_name)
  output_path = os.path.join(output_folder, output_name)

  raw_image_data = all_images_data[:, image_index].reshape((3, image_size, image_size))
  image_data = n.empty((image_size, image_size, 3), dtype=n.uint8)
  image_data[:, :, 0::3] = raw_image_data[0, :, :].reshape((image_size, image_size, 1))
  image_data[:, :, 1::3] = raw_image_data[1, :, :].reshape((image_size, image_size, 1))
  image_data[:, :, 2::3] = raw_image_data[2, :, :].reshape((image_size, image_size, 1))
  misc.imsave(output_path, image_data)
