#!/bin/sh

./tree_rings_benchmark.rb

gcc ../ext/tree_rings.c -o ../ext/tree_rings
../ext/tree_rings