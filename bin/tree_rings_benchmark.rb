#!/usr/bin/env ruby
#
# A baseline CPU-based benchmark program CPU/GPU performance comparision.
# Approximates the cross-sectional area of every tree ring in a tree trunk in serial and in parallel
# by taking the total area at a given radius and subtracting the area of the closest inner ring.
# Copyright © 2011 Preston Lee. All rights reserved.
# http://prestonlee.com
#

$:.unshift(File.join(File.dirname(__FILE__), '..', 'lib'))

require 'rubygems'
require	'ruby-gpu-examples'
require 'barracuda'

include Barracuda

RINGS = 1000000.to_i

NUM_THREADS = 8
threads = []

# Approximates the cross-sectional area of a tree ring by taking the total area at a given radius 
# and subtracting the area of the closest inner ring.
def ring_area(radius)
	(Math::PI * (radius ** 2)) - (Math::PI * ((radius - 1) ** 2))
end

def ring_job(rings, offset)
	last = offset + rings - 1
	(offset..(last)).each do |radius|
		ring_area(radius)
	end
end

# CUDA Setup
prog = Barracuda::Program.new
prog.compile <<EOF
	#define PI 3.141592653
  __kernel void ring_area(__global float * output) {
    int i = get_global_id(0);
    output[i] = (PI * i * i) - (PI * (i - 1) * (i - 1));	
  }
EOF

# Begin Processing
puts("\nA baseline CPU-based benchmark program for CPU/GPU performance comparision.")
puts("Copyright \u00A9 2011 Preston Lee. All rights reserved.\n\n")
puts("\tUsage: #{$0} [NUM TREE RINGS]\n\n")

puts("Number of tree rings: #{RINGS}. Yay!\n\n")

output = []

# Approximate the cross-sectional area between each pair of consecutive tree rings
# in serial
print("Running serial calculation using CPU...\t\t\t")
start = Time.now
(1..RINGS).each do |i|
	ring_area(i)
end
printf("%06f seconds\n", Time.now-start)

# Approximate the cross-sectional area between each pair of consecutive tree rings
# in parallel
print("Running parallel calculation using #{NUM_THREADS} CPU threads...\t")
start = Time.now
(0..(NUM_THREADS - 1)).each do |i|
	rings_per_thread = RINGS / NUM_THREADS
	offset = i * rings_per_thread
	threads << Thread.new(rings_per_thread, offset) do |num, offset|
		ring_job(num, offset)
	end
end
threads.each do |t| t.join end
printf("%06f seconds\n", Time.now-start)

# Approximate the cross-sectional area between each pair of consecutive tree rings
# on a GPU
print("Running parallel calculation using GPU...\t\t")
start = Time.now
output = Barracuda::Buffer.new(RINGS).to_type(:float)
prog.ring_area(output, :times => RINGS)
printf("%06f seconds\n\n", Time.now-start)
puts("Done!\n\n")