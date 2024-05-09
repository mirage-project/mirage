# Function for adding a test
function(add_mirage_test TEST_NAME TEST_ENTRY_PATH)
  cmake_parse_arguments(ADD_MIRAGE_TEST "" "ADDITIONAL_LIBS" "" ${ARGN})
  add_executable(${TEST_NAME} ${TEST_ENTRY_PATH})
  target_link_libraries(${TEST_NAME}
    gtest gtest_main mirage_runtime pthread)
  add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

add_subdirectory(threadblock)
add_subdirectory(search)
