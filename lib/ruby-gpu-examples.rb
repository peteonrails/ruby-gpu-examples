# Put native library on the load path...
$:.unshift(File.join(File.dirname(__FILE__), '..', 'ext'))

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

def printBanner(rings)
	# Begin Processing
	puts("\nA baseline CPU-based benchmark program for CPU/GPU performance comparision.")
	puts("Copyright \u00A9 2011 Preston Lee. All rights reserved.\n\n")
	puts("\tUsage: #{$0} [NUM TREE RINGS]\n\n")

	puts("Number of tree rings: #{rings}. Yay!\n\n")	
end


def runCPUSingleThreaded(rings)
	# Approximate the cross-sectional area between each pair of consecutive tree rings
	# in serial
	print("Running serial calculation using CPU...\t\t\t")
	start = Time.now
	(1..rings).each do |i|
		ring_area(i)
	end
	printf("%06f seconds\n", Time.now-start)	
end

def runCPUMultiThreaded(rings, num_threads = 4)
	# Approximate the cross-sectional area between each pair of consecutive tree rings
	# in parallel
	threads = []
	print("Running parallel calculation using #{num_threads} CPU threads...\t")
	start = Time.now
	(0..(num_threads - 1)).each do |i|
		rings_per_thread = rings / num_threads
		offset = i * rings_per_thread
		threads << Thread.new(rings_per_thread, offset) do |num, offset|
			ring_job(num, offset)
		end
	end
	threads.each do |t| t.join end
	printf("%06f seconds\n", Time.now-start)
end