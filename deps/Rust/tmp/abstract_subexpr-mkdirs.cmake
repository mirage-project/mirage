# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/xiaoyuj2/mirage/deps/Rust/src/abstract_subexpr")
  file(MAKE_DIRECTORY "/home/xiaoyuj2/mirage/deps/Rust/src/abstract_subexpr")
endif()
file(MAKE_DIRECTORY
  "/home/xiaoyuj2/mirage/src/search/abstract_expr/abstract_subexpr"
  "/home/xiaoyuj2/mirage/deps/Rust"
  "/home/xiaoyuj2/mirage/deps/Rust/tmp"
  "/home/xiaoyuj2/mirage/deps/Rust/src/abstract_subexpr-stamp"
  "/home/xiaoyuj2/mirage/deps/Rust/src"
  "/home/xiaoyuj2/mirage/deps/Rust/src/abstract_subexpr-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/xiaoyuj2/mirage/deps/Rust/src/abstract_subexpr-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/xiaoyuj2/mirage/deps/Rust/src/abstract_subexpr-stamp${cfgdir}") # cfgdir has leading slash
endif()
