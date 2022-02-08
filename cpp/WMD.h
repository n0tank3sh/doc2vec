#ifndef _DOC2VEC_WMD_H_
#define _DOC2VEC_WMD_H_

#include "common_define.h"

#include <limits>
#include <cstdio>

namespace doc2vec {
  class TaggedDocument;
  class WeightedDocument;
  class UnWeightedDocument;
  class Model;

  struct knn_item_t;

  class WMD {
  public:
    WMD(Model * doc2vec);
    ~WMD();
    
    void train();
    void save(FILE * fout) const;
    void load(FILE * fin);
    real rwmd(WeightedDocument * src, UnWeightedDocument * target);
    void sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, size_t k);
    void sent_knn_docs_ex(TaggedDocument & doc, knn_item_t * knns, size_t k);

  private:
    void loadFromDoc2Vec();

  public:
    UnWeightedDocument ** m_corpus;

    Model * m_doc2vec;
    knn_item_t * m_doc2vec_knns;
  };
};

#endif
