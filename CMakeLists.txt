if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.4)

project(sycl_popsift)


message(STATUS "Hello, SYCL POPSIFT -- could be used for different cmake compoilation message")

# This needs to be updated later on in addidtion to the whole thing to support GPU :D
set(SOURCE_FILE src/main.cpp) 
set(TARGET_NAME sycl_popsift)


# Find Boost libraries
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem program_options)

# Look for DevIL library
find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES, IL_INCLUDE_DIR

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Lists that we build and include in the target
set(PD_INCLUDE_DIRS    ${Boost_INCLUDE_DIRS})
set(PD_LINK_LIBS       ${Boost_LIBRARIES}) 


if(IL_FOUND OR DevIL_FOUND)
  message(STATUS "DevIL found, setting USE_DEVIL macro")
  set(PD_COMPILE_OPTIONS "-DUSE_DEVIL") # sets macro USE_DEVIL so we can use #ifdef in code

  # Add DevIL libraries to the target lists
  list(APPEND PD_INCLUDE_DIRS ${IL_INCLUDE_DIR})
  list(APPEND PD_LINK_LIBS    ${IL_LIBRARIES} ${ILU_LIBRARIES})
else()
  message(WARNING "DevIL not found -- Falling back to pgmread")
  set(PD_COMPILE_OPTIONS "" )
endif()



set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG} ${PD_COMPILE_OPTIONS}")
set(LINK_FLAGS "-fsycl")


add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

target_include_directories(${TARGET_NAME} PUBLIC ${PD_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${PD_LINK_LIBS})
