#include <limits>
#include "gtest/gtest.h"
#include <Model.h>
#include <common_define.h>

TEST(TestTrain, title_sg) {
  doc2vec::Model doc2vec;
  doc2vec::FileInput input("../data/paper.title.seg");
  doc2vec.train(input, 50, 0, 1, 0, 15, 10, 0.025, 1e-5, 3, 6);
  printf("\nWrite model to %s\n", "../data/model.title.sg");
  FILE * fout = fopen("../data/model.title.sg", "wb");
  doc2vec.save(fout);
  fclose(fout);
}
