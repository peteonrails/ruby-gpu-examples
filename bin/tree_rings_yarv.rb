#!/usr/bin/env ruby
#
# A baseline CPU-based benchmark program CPU/GPU performance comparision.
# Approximates the cross-sectional area of every tree ring in a tree trunk in serial and in parallel
# by taking the total area at a given radius and subtracting the area of the closest inner ring.
# Copyright © 2011 Preston Lee. All rights reserved.
# http://prestonlee.com
#

$:.unshift(File.join(File.dirname(__FILE__), '..', 'lib'))

require	'ruby-gpu-examples'
require	'ruby-gpu-examples/barracuda'

rings = 2**25.to_i
num_threads = 4

printBanner(rings)

runCPUSingleThreaded(rings)
runCPUMultiThreaded(rings, num_threads)
runGPU(rings)

puts("Done!\n\n")