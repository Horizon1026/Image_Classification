aux_source_directory( . AUX_MNIST_MLP )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_memory )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/memory ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_memory )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()

# Add all relative 3rd libraries.
if ( NOT TARGET lib_pytorch_for_cxx )
    add_subdirectory( ${SLAM_UTILITY_PATH}/3rd_libraries/libtorch ${PROJECT_SOURCE_DIR}/build/lib_pytorch_for_cxx )
endif()

add_library( lib_mnist_mlp ${AUX_MNIST_MLP} )
target_include_directories( lib_mnist_mlp PUBLIC . )
target_link_libraries( lib_mnist_mlp
    lib_slam_utility_basic_type
    lib_slam_utility_log
    lib_slam_utility_math
    lib_slam_utility_memory
    lib_slam_utility_operate
    lib_slam_utility_tick_tock
    lib_pytorch_for_cxx
)
