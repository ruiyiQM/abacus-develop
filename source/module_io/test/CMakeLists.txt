remove_definitions(-D__LCAO)
remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)

AddTest(
  TARGET input_test
  LIBS ${math_libs} base device
  SOURCES input_test.cpp ../input.cpp ../../src_parallel/parallel_reduce.cpp ../../src_parallel/parallel_global.cpp ../../src_parallel/parallel_common.cpp
)

install(FILES INPUT STRU DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET winput_test
  LIBS ${math_libs} base device
  SOURCES winput_test.cpp ../winput.cpp ../../src_parallel/parallel_reduce.cpp ../../src_parallel/parallel_global.cpp ../../src_parallel/parallel_common.cpp
)

install(FILES WINPUT DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET output_test
  LIBS ${math_libs} base device
  SOURCES output_test.cpp ../output.cpp ../../src_parallel/parallel_reduce.cpp ../../src_parallel/parallel_global.cpp ../../src_parallel/parallel_common.cpp
)

AddTest(
  TARGET rwstream_test
  SOURCES rwstream_test.cpp ../rwstream.cpp
)
