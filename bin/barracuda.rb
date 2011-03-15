#!/usr/bin/env ruby

$:.unshift(File.dirname(__FILE__) + '/../ext')

require 'barracuda'
require	'benchmark'


# CL - Enable these extensions for atom_add() on various types
EXTENSIONS = <<-'eof'
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
eof



puts "Defining GPU kernel..."
include Barracuda

program = Program.new <<EOF
	#{EXTENSIONS}
  __kernel void sum(__global int * input, __global int * output) {
    atom_add(output, input[get_global_id(0)]); 
  }
EOF

input		= (1..10).to_a
output	= Buffer.new(1)
# program.sum(input, output)


Benchmark.bm do |x|
	x.report('cpu')		{ puts input.inject(0) {|acc,n| acc + n } }
	x.report('cuda')	{ program.sum(input, output) }
end

puts "The sum is: " + output[0].to_s
