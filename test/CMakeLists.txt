enable_testing()
find_package(GTest REQUIRED)

set(SRC "test.cpp" "TestSimilar.cpp" "TestTrain.cpp")
add_executable(test ${SRC})
target_link_libraries(test GTest::gtest_main libdoc2vec)