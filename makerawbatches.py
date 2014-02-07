#!/usr/bin/python
#
# Copyright (c) 2011, Pete Warden
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

# This utility script takes a folder of images, named starting with the wordnet ID,
# and builds a set of raw label and image data batch files that can be read in
# by the LabeledRawDataProvider class.
#
# Usage is:
# python makerawbatches.py <image folder> <output batch folder> [label count]
#
# 'label count' is optional, but if it's set then only n ids will be used

import os
from scipy import misc
import re
import numpy as np
import pickle

IMAGES_PER_BATCH = 2500
IMAGE_SIZE = 256

log_counts = {}
def log_count(name, stride = 100):
  if name not in log_counts:
    log_counts[name] = 0
  log_counts[name] += 1
  if log_counts[name] % stride == 0:
    sys.stderr.write("%s %d\n" % (name, log_counts[name]))

if len(sys.argv) < 3:
  sys.stderr.write('Usage: python makerawbatches.py <image folder> <output batch folder> [label count]\n')
  exit(1)

image_folder = sys.argv[1]
output_folder = sys.argv[2]
if len(sys.argv) < 4:
  label_limit = None
else:
  label_limit = int(sys.argv[3])

found_ids = {}
input_image_glob = image_folder + '/*.jpg'
for index, image_path in enumerate(glob.iglob(input_image_glob)):
  if (index + 1) % 1000 == 0:
    sys.stderr.write('Found %d files\n' % (index + 1))
  basename = os.path.basename(image_path)
  try:
    id = re.search('^n([0-9]+)_', basename).group(1)
  except AttributeError:
    sys.stderr.write('No wordnet ID found for %s\n' % (image_path))
    continue
  if not id in found_ids:
    found_ids[id] = []
  found_ids[id].append(basename)

all_ids = found_ids.keys()

sys.stderr.write('Found %d total labels\n' % (len(all_ids)))

shuffled_ids = shuffle(all_ids)
if label_limit is None:
  wanted_ids = shuffled_ids
else:
  wanted_ids = shuffled_ids[0:label_limit]

sys.stderr.write('Looking for %d labels\n' % (len(wanted_ids)))

wanted_files = []
label_indexes = {}
for index, id in enumerate(wanted_ids):
  wanted_files += found_ids[id]
  label_indexes[id] = index

shuffled_wanted_files = shuffle(wanted_files)

sys.stderr.write('Starting to process %d files\n' % (len(wanted_files)))

total_image = np.zeros((IMAGE_SIZE * IMAGE_SIZE * 3), dtype=np.float64)

images_processed = 0
for i in xrange(0, len(wanted_files), IMAGES_PER_BATCH):
  current_images = shuffled_wanted_files[i:(i + IMAGES_PER_BATCH)]
  labels = []
  images = []
  for basename in current_images:
    image_path = image_folder + '/' + basename
    try:
      id = re.search('^n([0-9]+)_', basename).group(1)
    except AttributeError:
      sys.stderr.write('No wordnet ID found for %s\n' % (image_path))
      continue
    image = misc.imread(image_path)
    shape = image.shape
    width = shape[1]
    height = shape[0]
    channels = shape[2]
    if width == IMAGE_SIZE and height == IMAGE_SIZE:
      resized = image
    else:
      if width > height:
        margin = ((width - height) / 2)
        image = image[:, margin:-margin]
      if height > width:
        margin = ((height - width) / 2)
        image = image[margin:-margin, :]
      resized = misc.imresize(image, (IMAGE_SIZE, IMAGE_SIZE))
    red_channel = resized[:, :, 0]
    red_channel.shape = (IMAGE_SIZE * IMAGE_SIZE)
    green_channel = resized[:, :, 1]
    green_channel.shape = (IMAGE_SIZE * IMAGE_SIZE)
    blue_channel = resized[:, :, 2]
    blue_channel.shape = (IMAGE_SIZE * IMAGE_SIZE)
    all_channels = np.append(np.append(red_channel, green_channel), blue_channel)
    total_image += all_channels
    images.append(all_channels)
    label_index = label_indexes[id]
    labels.append(label_index)
    images_processed += 1
    log_count('Loaded', 100)

  images_data = np.vstack(images).transpose()
  labels_data = np.vstack(labels)
  output_index = (i / IMAGES_PER_BATCH)
  output_path= '%s/data_batch_%d' % (output_folder, output_index)
  output_file = open(output_path, 'wb')
  output_file.write(labels_data.tostring())
  output_file.write(images_data.tostring())
  output_file.close()
  sys.stderr.write('Wrote %s\n' % (output_path))

mean_image = total_image / images_processed
label_name_for_id = {}
# First, set up some default label names
for id in wanted_ids:
  label_name_for_id[id] = 'n' + str(id)
# Then, try to load them from the wordnet list
wordnet_lines = open('wordnetlabels.txt').read_lines()
for line in wordnet_lines:
  full_id, names = line.strip().split('\t')
  try:
    id = re.search('^n([0-9]+)', full_id).group(1)
  except AttributeError:
    sys.stderr.write('No wordnet ID found for %s\n' % (line.strip()))
    continue
  label_name_for_id[id] = names.split(',')[0].strip()
label_names = []
for id in wanted_ids:
  label_names.append(label_name_for_id[id])

meta = {'data_mean': mean_image, 'label_names': label_names}
meta_output_path= '%s/batches.meta' % (output_folder)
meta_output = open(meta_output_path, 'w')
pickle.dump(meta, meta_output)
meta_output.close()
