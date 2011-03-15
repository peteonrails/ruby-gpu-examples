#!/usr/bin/env ruby
#
# Approximates the cross-sectional area of every tree ring in a tree trunk in parallel
# by taking the total area at a given radius and subtracting the area of the closest inner ring.
#
# Author: Preston Lee
# 

$:.unshift(File.join(File.dirname(__FILE__), '..', 'lib'))

require	'ruby-gpu-examples'
require 'barracuda'
require 'benchmark'



rings = 2 ** 14




threads = []

# Approximates the cross-sectional area of a tree ring by taking the total area at a given radius and subtracting the area of the closest inner ring.
def ring_area(radius)
	(Math::PI * radius ** 2) - (Math::PI * (radius - 1) ** 2)
end

# puts area 2


prog = Barracuda::Program.new <<EOF
  __kernel void ring_area(__global float4 * output) {
    int i = get_global_id(0);
		
    if (i < total) output[i] = normalize(input[i]);
  }
EOF

Benchmark.bm do |x|
	x.report('CPU (Serial)') {
		(1..rings).each do |i|
			ring_area(i)
		end
	}
	
	x.report('CPU (Threads)') {
		# puts "Running #{rings} worker threads..."
		(1..rings).each do |i|
			threads << Thread.new do
				ring_area(i)
			end
		end
		# Thread.new

		# puts "Making sure every thread is done..."
		threads.each do |t| t.join end
		
	}
	x.report('GPU') {
		
	}
end


puts "Done!"