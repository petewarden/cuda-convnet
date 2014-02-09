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

# This utility script attempts to download labeled images from a list of URLs supplied by
# Imagenet: http://image-net.org/download-imageurls
# Only the Wordnet IDs listed in the second argument will be downloaded, there's an example
# for the ISLVRC 2011 competition in this folder as 'imagenet2011ids.txt'.
# You must supply an email address, so that website owners know who to contact in case the
# crawling process is causing them technical issues.
# The parallel threads parameter determines how many downloads happen at once. Because there's
# a lot of latency in download operations, setting this to 3 or 4 may help speed up the
# process, but start off at 1.
# The actual downloading is done by calling out to an external 'curl' shell command, so
# you'll need curl installed for this to work. I've taken this approach because I've found
# it's better at dealing with all the weird and wonderful error conditions and redirects
# you find in a large set of image URLs than any Python library.
# The images are downloaded to to the output folder, and the download is skipped for any
# files that are already present, so you can easily restart the process if it's interrupted.
#
# Usage is:
# python downloadimages.py <imagenet URLs file> <wordnets IDs file> <output folder> <your email address> <parallel threads>
#

import os
import sys
import threading
from subprocess import call

def log(message):
  sys.stderr.write("%s\n" % (message))

log_counts = {}
def log_count(name, stride = 1000):
  global log_counts
  if name not in log_counts:
    log_counts[name] = 0
  log_counts[name] += 1
  if log_counts[name] % stride == 0:
    log("%s %d" % (name, log_counts[name]))

def run(command):
  log('+ ' + command)
  try:
    return_code = call(command, shell=True)
    if return_code != 0:
      log('\'%s\' failed with return code %d' % (command, return_code))
  except OSError as e:
      log('\'%s\' failed with exception %s' % (command, str(e)))

def download_images(my_lines, output_folder, user_agent):
  for parts in my_lines:
    wordnet_full_id, photo_url = parts
    output_filename = os.path.join(output_folder, wordnet_full_id + '_' + os.path.basename(photo_url))
    if os.path.exists(output_filename):
      continue
    command = "curl -L '%s'" % (photo_url)
    command += " --user-agent '%s'" % (user_agent)
    command += " --fail"
    command += " --max-time 20"
    command += " -o '%s'" % (output_filename)
    run(command)

if len(sys.argv) < 6:
  log('Usage: python downloadimages.py <imagenet URLs file> <wordnets IDs file> <output folder> <your email address> <parallel threads>')
  exit(1)

urls_path = sys.argv[1]
wanted_path = sys.argv[2]
output_folder = sys.argv[3]
email_address = sys.argv[4]
thread_count = int(sys.argv[5])

user_agent = "Robot contact is '%s'" % email_address

log('Loading %s' % (wanted_path))
wanted_file = open(wanted_path)
wanted_text = wanted_file.read()
wanted_lines = wanted_text.split("\n")
wanted = {}
for line in wanted_lines:
  wordnet_id, names = line.strip().split("\t", 1)
  wanted[wordnet_id] = True

urls_file = open(urls_path)

input_lines = []
for thread_index in range(thread_count):
  input_lines.append([])

thread_index = 0
for line in urls_file:
  wordnet_full_id, photo_url = line.strip().split("\t", 1)
  wordnet_id, wordnet_index = wordnet_full_id.split('_', 1)
  if wordnet_id not in wanted:
    continue
  input_lines[thread_index].append([wordnet_full_id, photo_url])
  thread_index = ((thread_index + 1) % thread_count)
  log_count('read lines')

threads = []
for thread_index in range(thread_count):
  new_thread = threading.Thread(target=download_images,
    args = (input_lines[thread_index], output_folder, user_agent))
  new_thread.start()
  threads.append(new_thread)

for thread in threads:
  thread.join()

log("Finished processing")
