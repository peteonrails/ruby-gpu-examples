require 'barracuda'

include Barracuda


# Approximate the cross-sectional area between each pair of consecutive tree rings
# on a GPU
def runGPU(rings)
	require 'barracuda'
	# CUDA Setup
	prog = Barracuda::Program.new
	prog.compile <<EOF
		#define PI 3.141592653
	  __kernel void ring_area(__global float * output) {
	    int i = get_global_id(0);
	    output[i] = (PI * i * i) - (PI * (i - 1) * (i - 1));	
	  }
EOF


	output = []
	
	print("Running parallel calculation using GPU...\t\t")
	start = Time.now
	output = Barracuda::Buffer.new(rings).to_type(:float)
	prog.ring_area(output, :times => rings)
	printf("%06f seconds\n\n", Time.now-start)
	
end