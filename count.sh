#!/bin/bash

find . \( \
  -path "./build" -o \
  -path "./build-debug" -o \
  -path "./build-release" -o \
  -path "./debug-build" -o \
  -path "./release-build" -o \
  -path "./analysis" \) -prune \
  -o \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) -exec wc -l {} + | sort