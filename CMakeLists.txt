# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.25)

project(riva_asrlib LANGUAGES CUDA CXX)

include(FetchContent)

option(RIVA_ASRLIB_BUILD_PYTHON_BINDINGS "Whether to build python bindings." ON)



set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set (CMAKE_BUILD_TYPE Debug)
# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")

# set (CMAKE_BUILD_TYPE RelWithDebInfo)
# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread -fsanitize=undefined")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=thread -fsanitize=undefined")
# add_compile_definitions(_GLIBCXX_DEBUG)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

find_package(CUDAToolkit)

FetchContent_Declare(openfst
  URL http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz
  URL_HASH SHA256=21c3758ff8574dedaf22b0a6b88c2829bbf3b2e59fbf04740ce2cf92029a0f3b
  )

FetchContent_GetProperties(openfst)
if(NOT openfst_POPULATED)
  FetchContent_Populate(openfst)

  add_library(riva_asrlib_openfst
    ${openfst_SOURCE_DIR}/src/lib/properties.cc
    ${openfst_SOURCE_DIR}/src/lib/fst.cc
    ${openfst_SOURCE_DIR}/src/lib/util.cc
    ${openfst_SOURCE_DIR}/src/lib/compat.cc
    ${openfst_SOURCE_DIR}/src/lib/symbol-table.cc
    ${openfst_SOURCE_DIR}/src/lib/symbol-table-ops.cc
    ${openfst_SOURCE_DIR}/src/lib/flags.cc
    ${openfst_SOURCE_DIR}/src/lib/mapped-file.cc
    ${openfst_SOURCE_DIR}/src/lib/weight.cc
    )
  target_include_directories(riva_asrlib_openfst SYSTEM PUBLIC ${openfst_SOURCE_DIR}/src/include/)
  set_target_properties(riva_asrlib_openfst PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  target_compile_options(riva_asrlib_openfst PUBLIC -Wno-unused-local-typedefs -Wno-sign-compare -Wno-unused-variable)

  add_library(fst-types OBJECT "${openfst_SOURCE_DIR}/src/lib/fst-types.cc")
  target_include_directories(fst-types SYSTEM PUBLIC ${openfst_SOURCE_DIR}/src/include/)
  set_target_properties(fst-types PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  target_compile_options(fst-types PUBLIC -Wno-unused-local-typedefs -Wno-sign-compare -Wno-unused-variable)


  add_library(fstscript_base
    "${openfst_SOURCE_DIR}/src/script/arciterator-class.cc"
    "${openfst_SOURCE_DIR}/src/script/encodemapper-class.cc"
    "${openfst_SOURCE_DIR}/src/script/fst-class.cc"
    "${openfst_SOURCE_DIR}/src/script/getters.cc"
    "${openfst_SOURCE_DIR}/src/script/stateiterator-class.cc"
    "${openfst_SOURCE_DIR}/src/script/text-io.cc"
    "${openfst_SOURCE_DIR}/src/script/weight-class.cc"
    )
  target_link_libraries(fstscript_base PUBLIC riva_asrlib_openfst ${CMAKE_DL_LIBS} $<TARGET_OBJECTS:fst-types>)

  foreach(operation arcsort closure compile compose concat connect convert decode determinize disambiguate encode epsnormalize equal equivalent invert isomorphic map minimize project prune push randequivalent randgen relabel replace reverse reweight synchronize topsort union)
    # decode verify
    add_library("fstscript_${operation}" "${openfst_SOURCE_DIR}/src/script/${operation}.cc")
    target_link_libraries("fstscript_${operation}" PUBLIC fstscript_base)
    if (NOT ((${operation} STREQUAL decode) OR (${operation} STREQUAL randequivalent)))
      add_executable("fst${operation}" "${openfst_SOURCE_DIR}/src/bin/fst${operation}.cc" "${openfst_SOURCE_DIR}/src/bin/fst${operation}-main.cc")
      target_link_libraries("fst${operation}" "fstscript_${operation}")
    endif()
  endforeach()

  target_link_libraries(fstscript_encode PUBLIC fstscript_decode)
  target_link_libraries(fstscript_equivalent PUBLIC fstscript_randequivalent)
endif()

# Created with `ls -1 third_party/kaldi/src/{base,util,matrix,fstext,lm,lat}/*.cc | grep -v test`
set(cuda_decoder_sources
  third_party/kaldi/src/base/io-funcs.cc
  third_party/kaldi/src/base/kaldi-error.cc
  third_party/kaldi/src/base/kaldi-math.cc
  third_party/kaldi/src/base/kaldi-utils.cc
  third_party/kaldi/src/base/timer.cc
  third_party/kaldi/src/fstext/context-fst.cc
  third_party/kaldi/src/fstext/grammar-context-fst.cc
  third_party/kaldi/src/fstext/kaldi-fst-io.cc
  third_party/kaldi/src/fstext/push-special.cc
  third_party/kaldi/src/lat/compose-lattice-pruned.cc
  third_party/kaldi/src/lat/confidence.cc
  third_party/kaldi/src/lat/determinize-lattice-pruned.cc
  third_party/kaldi/src/lat/kaldi-lattice.cc
  third_party/kaldi/src/lat/lattice-functions.cc
  # third_party/kaldi/src/lat/lattice-functions-transition-model.cc
  third_party/kaldi/src/lat/minimize-lattice.cc
  third_party/kaldi/src/lat/phone-align-lattice.cc
  third_party/kaldi/src/lat/push-lattice.cc
  third_party/kaldi/src/lat/sausages.cc
  third_party/kaldi/src/lat/word-align-lattice.cc
  third_party/kaldi/src/lat/word-align-lattice-lexicon.cc
  third_party/kaldi/src/lm/arpa-file-parser.cc
  third_party/kaldi/src/lm/arpa-lm-compiler.cc
  third_party/kaldi/src/lm/const-arpa-lm.cc
  third_party/kaldi/src/lm/kaldi-rnnlm.cc
  third_party/kaldi/src/lm/kenlm.cc
  third_party/kaldi/src/lm/mikolov-rnnlm-lib.cc
  third_party/kaldi/src/matrix/compressed-matrix.cc
  third_party/kaldi/src/matrix/kaldi-matrix.cc
  third_party/kaldi/src/matrix/kaldi-vector.cc
  third_party/kaldi/src/matrix/matrix-functions.cc
  third_party/kaldi/src/matrix/numpy-array.cc
  third_party/kaldi/src/matrix/optimization.cc
  third_party/kaldi/src/matrix/packed-matrix.cc
  third_party/kaldi/src/matrix/qr.cc
  third_party/kaldi/src/matrix/sparse-matrix.cc
  third_party/kaldi/src/matrix/sp-matrix.cc
  third_party/kaldi/src/matrix/srfft.cc
  third_party/kaldi/src/matrix/tp-matrix.cc
  third_party/kaldi/src/util/kaldi-holder.cc
  third_party/kaldi/src/util/kaldi-io.cc
  third_party/kaldi/src/util/kaldi-semaphore.cc
  third_party/kaldi/src/util/kaldi-table.cc
  third_party/kaldi/src/util/kaldi-thread.cc
  third_party/kaldi/src/util/parse-options.cc
  third_party/kaldi/src/util/simple-io-funcs.cc
  third_party/kaldi/src/util/simple-options.cc
  third_party/kaldi/src/util/text-utils.cc
  third_party/kaldi/src/decoder/grammar-fst.cc
  third_party/kaldi/src/decoder/lattice-faster-decoder.cc
  third_party/kaldi/src/decoder/lattice-faster-online-decoder.cc
  third_party/kaldi/src/decoder/lattice-incremental-decoder.cc
  third_party/kaldi/src/decoder/lattice-incremental-online-decoder.cc
  third_party/kaldi/src/cudadecoder/batched-static-nnet3-kernels.cu
  third_party/kaldi/src/cudadecoder/cuda-decoder.cc
  third_party/kaldi/src/cudadecoder/cuda-decoder-kernels.cu
  third_party/kaldi/src/cudadecoder/lattice-postprocessor.cc
  third_party/kaldi/src/cudadecoder/cuda-fst.cc
  third_party/kaldi/src/cudadecoder/thread-pool-cia.cc
  third_party/kaldi/src/online2/online-endpoint.cc

  src/riva/asrlib/decoder/ctc_transition_information.cc
  src/riva/asrlib/decoder/batched-mapped-decoder-cuda.cc
  )

add_library(riva_asrlib_wfst_decoder
  ${cuda_decoder_sources}
  )

set_target_properties(riva_asrlib_wfst_decoder PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(riva_asrlib_wfst_decoder
  # This include directory must appear first so that the correct
  # matrix/cblas-wrappers.h is picked up
  PUBLIC "include/"
  PUBLIC "third_party/kaldi/src/"
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(riva_asrlib_wfst_decoder
  PUBLIC HAVE_CUDA=1)
target_compile_features(riva_asrlib_wfst_decoder PUBLIC cxx_std_17
  PUBLIC cuda_std_17)
target_link_libraries(riva_asrlib_wfst_decoder
  PUBLIC Threads::Threads
  PRIVATE CUDA::nvtx3
  PUBLIC riva_asrlib_openfst)
# target_link_options(riva_asrlib_wfst_decoder PRIVATE -Wl,--no-undefined)
target_compile_options(riva_asrlib_wfst_decoder PUBLIC -Wno-unused-but-set-variable -ffunction-sections -fdata-sections)

add_executable(offline-cuda-decode-binary src/riva/asrlib/decoder/bin/offline-cuda-decode-binary.cc)
target_link_libraries(offline-cuda-decode-binary PUBLIC riva_asrlib_wfst_decoder
  PUBLIC CUDA::cudart_static)

foreach(fstext_bin_name fstdeterminizestar fstrmsymbols fstisstochastic fstminimizeencoded fstmakecontextfst fstmakecontextsyms fstaddsubsequentialloop fstaddselfloops fstrmepslocal fstcomposecontext fsttablecompose fstrand fstdeterminizelog fstphicompose fstcopy fstpushspecial fsts-to-transcripts fsts-project fsts-union fsts-concat)
  add_executable(${fstext_bin_name} "third_party/kaldi/src/fstbin/${fstext_bin_name}.cc")
  target_link_libraries(${fstext_bin_name} riva_asrlib_wfst_decoder)
endforeach()

add_executable(transcripts-to-fsts "third_party/kaldi/src/kwsbin/transcripts-to-fsts.cc")
target_link_libraries(transcripts-to-fsts riva_asrlib_wfst_decoder)

add_executable(arpa2fst "third_party/kaldi/src/lmbin/arpa2fst.cc")
target_link_libraries(arpa2fst riva_asrlib_wfst_decoder)
add_executable(arpa-to-const-arpa "third_party/kaldi/src/lmbin/arpa-to-const-arpa.cc")
target_link_libraries(arpa-to-const-arpa riva_asrlib_wfst_decoder)

# add_test(NAME offline-cuda-decode-binary-test
#   COMMAND ${CMAKE_SOURCE_DIR}/src/offline-cuda-decode-binary-test.sh --config $<CONFIG> --exe $<TARGET_FILE:offline-cuda-decode-binary>)

if (RIVA_ASRLIB_BUILD_PYTHON_BINDINGS)
  add_subdirectory(third_party/pybind11)
  add_subdirectory(third_party/dlpack)
  pybind11_add_module(python_decoder src/riva/asrlib/decoder/python_decoder.cc)
  target_link_libraries(python_decoder PUBLIC riva_asrlib_wfst_decoder dlpack)
endif(RIVA_ASRLIB_BUILD_PYTHON_BINDINGS)
