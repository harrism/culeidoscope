set(LLVM_LINK_COMPONENTS core jit interpreter native bitwriter)
set(LLVM_REQUIRES_RTTI 1)

#Searching CUDA
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set( NVVM_BASE C:/src/libNVVM/lib/ )

if (WIN32)
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( NVVM_HOME ${NVVM_BASE}/x86_64_win32_release/ )   
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 ${NVVM_HOME}) 
  else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( NVVM_HOME ${NVVM_BASE}/i686_win32_release/ )   
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32 ${NVVM_HOME}) 
  endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
else (WIN32) # todo add support for Mac
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( NVVM_HOME ${NVVM_BASE}/x86_64_Linux_release/ )   
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x86_64 ${NVVM_HOME}) 
  else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( NVVM_HOME ${NVVM_BASE}/i686_Linux_release/ )   
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/i686 ${NVVM_HOME}) 
  endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
endif (WIN32)

message(STATUS ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS ${NVVM_HOME})

add_llvm_example(culeidoscope
  toy.cpp
  vector.cpp
  vector.h
  nvvmwrapper.cpp
  nvvmwrapper.h
  launch.cpp
  drvapi_error_string.h
  )

target_link_libraries(culeidoscope cuda.lib nvvm.lib)