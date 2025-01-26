add_library(sycl_popsift
            sycl_popsift/popsift.cpp sycl_popsift/popsift.hpp
            sycl_popsift/s_image.cpp sycl_popsift/s_image.hpp
            sycl_popsift/sift_pyramid.cpp sycl_popsift/sift_pyramid.hpp
            sycl_popsift/s_pyramid_build.cpp sycl_popsift/s_pyramid_bulid_ra.cpp
            sycl_popsift/sift_constants.cpp sycl_popsift/sift_constants.cpp
            sycl_popsift/sift_octave.cpp sycl_popsift/sift_octave.hpp
            sycl_popsift/gauss_filter.cpp sycl_popsift/gauss_filter.cpp
            sycl_popsift/non_sycl/sift_conf.cpp sycl_popsift/non_sycl/sift_conf.hpp
            sycl_popsift/common/debug_macros.cpp sycl_popsift/common/debug_macros.hpp
            sycl_popsift/common/sync_queue.h
            sycl_popsift/common/assist.h
      )

# TODO: Figure out if it should be STATIC, SHARED or not set like in popsift (aka using BUILD_SHARED_LIBS to do
# do it differntly for MSVC and shared libs

add_sycl_to_target(TARGET sycl_popsift SOURCES
  sycl_popsift/popsift.cpp sycl_popsift/popsift.hpp
  sycl_popsift/s_image.cpp sycl_popsift/s_image.hpp
  sycl_popsift/sift_pyramid.cpp sycl_popsift/sift_pyramid.hpp
  sycl_popsift/s_pyramid_build.cpp sycl_popsift/s_pyramid_bulid_ra.cpp
  sycl_popsift/sift_constants.cpp sycl_popsift/sift_constants.cpp
  sycl_popsift/sift_octave.cpp sycl_popsift/sift_octave.hpp
  sycl_popsift/gauss_filter.cpp sycl_popsift/gauss_filter.cpp
)


target_link_libraries(sycl_popsift
                      PUBLIC
                      Threads::Threads
                    )

# build directory containing the automatically generated files
set(popsift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# BUILD_INTERFACE allows to include the directory with source only when target is
# built in the building tree (ie, not from an install location)
target_include_directories(sycl_popsift
            PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${popsift_generated_dir}>
            $<BUILD_INTERFACE:${popsift_generated_dir}/sycl_popsift>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
	    )

# TODO: Add install target like popsift

if(PopSift_BUILD_EXAMPLES)
  message(STATUS "Building demo!")
  add_subdirectory(application)
else()
  message(STATUS "Not building demo!")
endif()







# ERRORNOUS CODE BELOW


# # included to try to fix cmake issue that it can't determine language
# # set_target_properties(popsift PROPERTIES LINKER_LANGUAGE CXX)
#
# # if(PopSift_USE_NVTX_PROFILING)
# # target_link_libraries(popsift
# #                       PUBLIC
# # 		      CUDA::nvtx3)
# # endif()
#
# # TODO: HAve project version in version.hpp like popsift
# set_target_properties(sycl_popsift PROPERTIES VERSION ${PROJECT_VERSION})
# # set_target_properties(popsift PROPERTIES DEBUG_POSTFIX "d")
# # set_target_properties(popsift PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#
# # build directory containing the automatically generated files
# # set(popsift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
#
# # BUILD_INTERFACE allows to include the directory with source only when target is
# # built in the building tree (ie, not from an install location)
# # The CUDA install dir variable has changed from the old CUDA_INCLUDE_DIRS to the new CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
# target_include_directories(sycl_popsift
#             PUBLIC
#             $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
#             $<BUILD_INTERFACE:${popsift_generated_dir}>
#             $<BUILD_INTERFACE:${popsift_generated_dir}/popsift>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
# 	    # ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# 	    )
#
# # EXPORTING THE LIBRARY
# #
# # place to put the cmake-related files
# set(popsift_config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
# # include directory for install
# set(popsift_include_install_dir "${CMAKE_INSTALL_INCLUDEDIR}")
# # the name for the generated header version file
# set(popsift_version_header_name "${popsift_generated_dir}/sycl_popsift/version.hpp")
# # the name for the generated config.hpp
# set(popsift_config_header_name "${popsift_generated_dir}/sycl_popsift/sift_config.h")
# # where to install the generated files
# set(popsift_install_dir_generated "${popsift_include_install_dir}/sycl_popsift")
#
#
# # Configuration
# set(popsift_cmake_version_config "${popsift_generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
# set(popsift_cmake_project_config "${popsift_generated_dir}/${PROJECT_NAME}Config.cmake")
# set(popsift_targets_export_name "${PROJECT_NAME}Targets")
# set(popsift_namespace "${PROJECT_NAME}::")
#
# # Include module with fuction 'write_basic_package_version_file'
# include(CMakePackageConfigHelpers)
#
# # Configure '<PROJECT-NAME>ConfigVersion.cmake'
# # Note: major version number must be the same as requested
# # write_basic_package_version_file("${popsift_cmake_version_config}" COMPATIBILITY SameMajorVersion)
#
# # Configure '<PROJECT-NAME>Config.cmake'
# # Use variables:
# #   * popsift_targets_export_name
# #   * PROJECT_NAME
# configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
#                               "${popsift_cmake_project_config}"
#                               INSTALL_DESTINATION "${popsift_config_install_dir}")
#
# # version file
# configure_file("${PROJECT_SOURCE_DIR}/cmake/version.hpp.in" ${popsift_version_header_name} @ONLY)
#
# # TODO: Use config file like this in future -- if multiple options exists in the code base
#
# # configure_file("${PROJECT_SOURCE_DIR}/cmake/sift_config.h.in" ${popsift_config_header_name} @ONLY)
#
#
# # Targets:
# #   * <prefix>/lib/libpopsift.a
# #   * header location after install: <prefix>/include/
# #   * headers can be included by C++ code `#include <popsift/popsift.h>`
# install(TARGETS sycl_popsift
#         EXPORT "${popsift_targets_export_name}"
#         LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#         ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#         RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
#         INCLUDES DESTINATION "${popsift_include_install_dir}")
#
# # Headers:
# install(DIRECTORY "popsift"
#         DESTINATION "${popsift_include_install_dir}"
#         FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
#
# install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/popsift"
#         DESTINATION "${popsift_include_install_dir}"
#         FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
#
# # Config
# #   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
# #   * <prefix>/lib/cmake/${PROJECT_NAME}${PROJECT_NAME}ConfigVersion.cmake
# install(FILES "${popsift_cmake_project_config}" "${popsift_cmake_version_config}"
#         DESTINATION "${popsift_config_install_dir}")
#
# # Config
# #   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake
# install(EXPORT "${popsift_targets_export_name}"
#         NAMESPACE "${popsift_namespace}"
#         DESTINATION "${popsift_config_install_dir}")
#
# # install the config and version generated files
# install( FILES "${popsift_config_header_name}"
# 		DESTINATION "${popsift_install_dir_generated}")
#
# install( FILES "${popsift_version_header_name}"
# 		DESTINATION "${popsift_install_dir_generated}")
#
# # Generate ${PROJECT_NAME}Targets.cmake in the build directory to use the library without installing it
# export(TARGETS sycl_popsift FILE "${popsift_generated_dir}/${popsift_targets_export_name}.cmake")
#
