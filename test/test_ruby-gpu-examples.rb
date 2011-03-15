require 'helper'

class TestRubyGpuExamples < Test::Unit::TestCase

	should "load barracuda gem correctly" do
		begin
			require 'barracuda'
		rescue
			flunk "library failed to load"
		end
	end
	
end
