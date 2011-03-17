#!/usr/bin/env ruby
#
# A JRuby-friendly, CPU-based benchmark program CPU/GPU performance comparision.
# Approximates the cross-sectional area of every tree ring in a tree trunk in serial and in parallel
# by taking the total area at a given radius and subtracting the area of the closest inner ring
# in both single and muilti-threaded fashion.
#
# Copyright © 2011 Preston Lee. All rights reserved.
# http://prestonlee.com
#

$:.unshift(File.join(File.dirname(__FILE__), '..', 'lib'))

require	'ruby-gpu-examples'

rings = 2**25.to_i
num_threads = 4

printBanner(rings)

runCPUSingleThreaded(rings)
runCPUMultiThreaded(rings, num_threads)

puts("Done!\n\n")
