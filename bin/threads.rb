#!/usr/bin/env ruby
#
# A brief demonstration of the inherent problem of running multiple threads
# on modern SMP (Semetric Multi-Processing, aka multi-core) machines.
#
# This will use native threads in Ruby 1.9, but only run on one CPU core
# due to the stupid Global Interpreter Lock (GIL). Use a tool such as
# Activity Monitor.app (OS X), `top' (Linux) or Task Manager (Windows)
# to see the CPU usage of this script during execution.
#
# See: http://yehudakatz.com/2010/08/14/threads-in-ruby-enough-already/
#
# Author: Preston Lee

THREADS = 8

puts "Creating #{THREADS} worker threads."
threads = []
(1..THREADS).each do |n|
	threads << Thread.new(n) do |n|
		# Do some long-running computation here...
		(1..1e7.to_i).each do |x|
			Math::PI * x # Math is fun.. or something!
		end
	end
end

puts "Waiting for workers to finish."
threads.each do |t| t.join end

puts "Done!"
