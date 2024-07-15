set(SRC 
  "Model.cpp"
  "NN.cpp"
  "Vocabulary.cpp"
  "TrainModelThread.cpp"
  "TaggedBrownCorpus.cpp"
  "WMD.cpp"
  )

add_library(libdoc2vec ${SRC})
target_include_directories(libdoc2vec PUBLIC ../include/)
add_executable(train "train.cpp")
target_link_libraries(train libdoc2vec)
