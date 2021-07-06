cmake_minimum_required(VERSION 2.8)
project( pqtree)
find_package( OpenCV REQUIRED )
find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)

#set(GCC_FLAGS
#"-g -ggdb -O3 -m64 -fopenmp")

include_directories( ${OpenCV_INCLUDE_DIRS} ${BOOST_INCLUDEDIR} )
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -fopenmp -w")


add_executable( pqtree main.cpp pq.cpp utils.cpp pq_tree.cpp pqbase.cpp) # find_edges.h required
target_link_libraries( pqtree ${OpenCV_LIBS} )

add_executable( deltapqtree deltapq_approx_tree_main.cpp pq.cpp utils.cpp pq_tree.cpp)
target_link_libraries( deltapqtree ${OpenCV_LIBS} )
