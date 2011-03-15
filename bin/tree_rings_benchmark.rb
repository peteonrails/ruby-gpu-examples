#!/usr/bin/env ruby
#
# Approximates the cross-sectional area of every tree ring in a tree trunk in parallel
# by taking the total area at a given radius and subtracting the area of the closest inner ring.
#
# Author: Preston Lee
# 

$:.unshift(File.join(File.dirname(__FILE__), '..', 'lib'))

require 'rubygems'
require	'ruby-gpu-examples'
require 'barracuda'
require 'benchmark'



RINGS = 1e3.to_i



NUM_THREADS = 8
threads = []

# Approximates the cross-sectional area of a tree ring by taking the total area at a given radius and subtracting the area of the closest inner ring.
def ring_area(radius)
	(Math::PI * (radius ** 2)) - (Math::PI * ((radius - 1) ** 2))
end

def ring_job(rings, offset)
	last = offset + rings - 1
	puts "Computing rings #{offset} - #{last}."
	(offset..(last)).each do |radius|
		ring_area(radius)
	end
end


prog = Barracuda::Program.new <<EOF
#define PI 3.141592653

  __kernel void ring_area(__global float4 * output) {
    int i = get_global_id(0);
    output[i] = (PI * i * i) - (PI * (i - 1) * (i - 1));		
  }
EOF

puts "Rings: #{RINGS}. Yay!"

Benchmark.bm do |x|
	# x.report("CPU (Serial)\t") {
	# 	(1..RINGS).each do |i|
	# 		ring_area(i)
	# 	end
	# }
	
	x.report("CPU (#{NUM_THREADS} Threads)\t") {
		# puts "Running #{rings} worker threads..."
		(0..(NUM_THREADS - 1)).each do |i|
			# puts "Creating thread #{i}."
			rings_per_thread = RINGS / NUM_THREADS
			offset = i * rings_per_thread
			
			threads << Thread.new(rings_per_thread, offset) do |num, offset|
				ring_job(num, offset)
			end
		end
		# Thread.new

		# puts "Making sure every thread is done..."
		threads.each do |t| t.join end
		
	}
	x.report('GPU') {
		output = Barracuda::Buffer.new(RINGS).to_type(:float)
		puts prog.ring_area(output, :times => RINGS)
	}
end


puts "Done!"