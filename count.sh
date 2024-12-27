#!/bin/bash

find . \( -path "./build-debug" -o -path "./build-release" \) -prune \
  -o \( -name "*.cpp" -o -name "*.h" \) -exec wc -l {} + | sort