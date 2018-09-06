#
# CMakeLists.txt  cmake for mpibarrier-runner
#

#
# Use CMAKE_INSTALL_PATH or CMAKE_PREFIX_PATH to specify custom lib locations
#
cmake_minimum_required(VERSION 3.1)

project (mpibarrier-runner C CXX)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (CMAKE_CXX_STANDARD 11)

find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

set (CMAKE_THREAD_PREFER_PTHREAD TRUE)
set (THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package (Threads REQUIRED)

include (CheckCXXCompilerFlag)
check_cxx_compiler_flag (-Wall flagWall)
if (flagWall)
    add_compile_options (-Wall)
endif ()

add_executable (mpibarrier-runner mpibarrier-runner.cc)
target_link_libraries (mpibarrier-runner Threads::Threads)

# XXX: MPICH on Ubuntu gives a leading space that we need to trim off
foreach (lcv ${MPI_CXX_COMPILE_FLAGS_LIST})
    if (NOT ${lcv} STREQUAL "")
        target_compile_options (mpibarrier-runner PUBLIC $<BUILD_INTERFACE:${lcv}>)
    endif ()
endforeach ()

foreach (lcv ${MPI_CXX_INCLUDE_PATH})
    target_include_directories (mpibarrier-runner PUBLIC $<BUILD_INTERFACE:${lcv}>)
endforeach ()

foreach (lcv ${MPI_CXX_LIBRARIES})
    target_link_libraries(mpibarrier-runner $<BUILD_INTERFACE:${lcv}>)
endforeach ()

set_property (TARGET mpibarrier-runner APPEND PROPERTY LINK_FLAGS
        ${MPI_CXX_LINK_FLAGS})

#
# "make install" rule
#
install (TARGETS mpibarrier-runner RUNTIME DESTINATION bin)
