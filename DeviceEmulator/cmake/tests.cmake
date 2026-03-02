# add_executable(test_torch_layout test_torch_layout.cpp)
# target_include_directories(test_torch_layout PRIVATE
# ${TORCH_INCLUDE_DIRS}
# )
# target_link_libraries(test_torch_layout ${TORCH_LIBRARIES})
# set_target_properties(test_torch_layout PROPERTIES
# RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
# )
# add_test(NAME test_torch_layout COMMAND test_torch_layout)



#
# Define a target named `TEST_NAME` for a single extension.
# Optional arguments:
#
# LIBRARIES <libraries>      - Extra link libraries.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# SOURCES <sources>           - Extra source files.
#
function(add_test_executable TEST_NAME)
    cmake_parse_arguments(TEST "" "LIBRARIES;INCLUDE_DIRECTORIES;SOURCES" "" ${ARGN})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp ${TEST_SOURCES})
    target_include_directories(${TEST_NAME} PRIVATE
        ${TEST_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${TEST_NAME}
        ${TEST_LIBRARIES}
    )
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()
