#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script builds a TFLite micro test binary and compare the size difference
# between this binary and that same binary from the main repo.
# If the optional argument string "error_on_memory_increase" is provided as the
# script input, the script will error exit on any memory increase.
# If no argument is provided, the script produce a size comparison report.
# In addition to the optional "error_on_memory_increase", the rest of optional
# parameters are for Makefile and reflect make syntax. An example is
#
# test_size.sh error_on_memory_increase  BUILD_TYPE=release
# TARGET_ARCH=TARGET=xtensa TARGET_ARCH=hifi4 OPTIMIZED_KERNEL_DIR=xtensa
# XTENSA_CORE=F1_190305_swupgrade 
#
set -e

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# Utility function to build a target and return its path back to caller through
# a global variable __BINARY_TARGET_PATH.
# The caller is expected to store this  __BINARY_TARGET_PATH back to its local
# variable if it needs to use the generated binary target with path later on.
__BINARY_TARGET_PATH=
function build_target() {
  local make_args=$@
  local binary_target=$1
  readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile build ${make_args}

  # Return the relative binary with path and name.
  local build_type=default
  local target=linux
  local target_arch=x86_64
  for arg in ${make_args};
  do
    if [[ "${arg}" =~ ^"BUILD_TYPE=" ]];
    then
      build_type=$(expr substr ${arg} 12 30)
    elif [[ "${arg}" =~ ^"TARGET=" ]];
    then
      target=$(expr substr ${arg} 8 30)
    elif [[ "${arg}" =~ ^"TARGET_ARCH=" ]];
    then
      target_arch=$(expr substr ${arg} 13 30)
    fi
  done
  __BINARY_TARGET_PATH="tensorflow/lite/micro/tools/make/gen/${target}_${target_arch}_${build_type}/bin/${binary_target}"
}

# Global flags
FLAG_ERROR_ON_MEM_INCREASE=
# Make file flags are just pass in.
MAKEFLAGS=

# Parse input arguments. Cannot use getopt because most parameters
# are just passin to Makefile.
for arg in $@
do
  if [ "${arg}" = "error_on_mem_increase" ];
  then 
    FLAG_ERROR_ON_MEM_INCREASE=${arg}
  else
    MAKEFLAGS="${MAKEFLAGS} ${arg}"
  fi
done

# Print out the configuration for better on target support
echo FLAG_ERROR_ON_MEM_INCREASE is ${FLAG_ERROR_ON_MEM_INCREASE}
echo other makefile flags are ${OTHER_MAKEFLAGS}

# Pick keyword_benchmark as the target
BENCHMARK_TARGET=keyword_benchmark

# Get the root directory of the current repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..

# Build a binary for the current repo
cd "${ROOT_DIR}"
# Clean once.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

build_target ${BENCHMARK_TARGET} ${MAKEFLAGS}
CURRENT_BINARY=${__BINARY_TARGET_PATH}
size ${CURRENT_BINARY} > ${ROOT_DIR}/ci/size_log.txt

# Get a clone of the main repo as the reference.
# This is nested in the current repo because in the case that we need to
# run docker, only the current repo root is mounted.
REF_ROOT_DIR="$(mktemp -d ${ROOT_DIR}/main_ref.XXXXXX)"
git clone https://github.com/tensorflow/tflite-micro.git  ${REF_ROOT_DIR}

# Build a binary for the main repo.
cd ${REF_ROOT_DIR}
build_target ${BENCHMARK_TARGET} ${MAKEFLAGS}
REF_BINARY=${__BINARY_TARGET_PATH}
size ${REF_BINARY} > ${REF_ROOT_DIR}/ci/size_log.txt

# Compare the two files at th root of current repo.
cd ${ROOT_DIR}
if [ "${FLAG_ERROR_ON_MEM_INCREASE}" = "error_on_mem_increase" ]
then
  tensorflow/lite/micro/tools/ci_build/size_comp.py -a ${REF_ROOT_DIR}/ci/size_log.txt ${ROOT_DIR}/ci/size_log.txt --error_on_mem_increase
else
  tensorflow/lite/micro/tools/ci_build/size_comp.py -a ${REF_ROOT_DIR}/ci/size_log.txt ${ROOT_DIR}/ci/size_log.txt
fi
