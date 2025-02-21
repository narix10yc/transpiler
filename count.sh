#!/bin/bash

find . \( \
  -path "./build" -o \
  -path "./build-debug" -o \
  -path "./build-release" -o \
  -path "./analysis" \) -prune \
  -o \( -name "*.cpp" -o -name "*.h" \) -exec wc -l {} + | sort