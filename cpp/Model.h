#ifndef _DOC2VEC_MODEL_H_
#define _DOC2VEC_MODEL_H_

#include "Vocabulary.h"
#include "NN.h"
#include "WMD.h"
#include "TaggedBrownCorpus.h"

#include "common_define.h"

#include <vector>
#include <string>
#include <memory>

namespace doc2vec {
  class TrainModelThread;
  class NN;
  class Vocabulary;
  class WMD;
  class TaggedBrownCorpus;
  class TaggedDocument;
  
  struct knn_item_t;

  class Model {
    friend class TrainModelThread;  
  public:
    Model();
  
    void train(const std::string & train_file,
	       size_t dim, bool cbow, bool hs, int negative,
	       int iter, int window,
	       real alpha, real sample,
	       int min_count, int threads);

    size_t dim() const;
    WMD & wmd() { return *m_wmd; }
    const Vocabulary & wvocab() const { return *m_word_vocab; }
    const Vocabulary & dvocab() const { return *m_doc_vocab; }
    NN & nn() { return *m_nn; };
    TaggedBrownCorpus & brownCorpus() { return *m_brown_corpus; }

    real doc_likelihood(TaggedDocument & doc, int skip = -1);
    real context_likelihood(TaggedDocument & doc, int sentence_position);
    void infer_doc(TaggedDocument & doc, real * vec, int skip = -1);
    bool word_knn_words(const std::string & search, knn_item_t * knns, size_t k);
    bool doc_knn_docs(const std::string & search, knn_item_t * knns, size_t k);
    bool word_knn_docs(const std::string & search, knn_item_t * knns, size_t k);

    void sent_knn_words(TaggedDocument & doc, knn_item_t * knns, size_t k, real * infer_vector);
    void sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, size_t k, real * infer_vector);

    void sent_knn_words(TaggedDocument & doc, knn_item_t * knns, size_t k);
    void sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, size_t k);
  
    real similarity(const real * src, const real * target) const;
    real distance(const real * src, const real * target) const;

    void save(FILE * fout) const;
    void load(FILE * fin);

    real getStartAlpha() const { return m_start_alpha; }
    size_t iter() const { return m_iter; }
    real getAlpha() const { return m_alpha; }
    void setAlpha(float a) { m_alpha = a; }
    void updateWordCountActual(long long d) { m_word_count_actual += d; }
    bool useHS() const { return m_hs; }
    int negative() const { return m_negative; }
    const int * getNegativeSampleTable() const { return m_negative_sample_table.get(); }

  private:
    void initExpTable();
    void initNegTable();
    void initTrainModelThreads(const std::string & train_file, int threads, int iter, std::vector<TrainModelThread *> & trainModelThreads);
    bool obj_knn_objs(const std::string & search, const real * src,
		      bool search_is_word, bool target_is_word,
		      knn_item_t * knns, size_t k);

    std::unique_ptr<Vocabulary> m_word_vocab;
    std::unique_ptr<Vocabulary> m_doc_vocab;
    std::unique_ptr<NN> m_nn;
    std::unique_ptr<WMD> m_wmd;
  
    bool m_cbow = true;
    bool m_hs;
    int m_negative;
    int m_window;
    real m_start_alpha; //fix lr
    real m_sample;
    int m_iter;

    //no need to flush to disk
    std::unique_ptr<TaggedBrownCorpus> m_brown_corpus;
    real m_alpha; //working lr
    long long m_word_count_actual;
    std::unique_ptr<real[]> m_expTable;
    std::unique_ptr<int[]> m_negative_sample_table;
  };

  struct knn_item_t
  {
    std::string word;
    long long idx;
    real similarity;
  };
  void top_init(knn_item_t * knns, size_t k);
  void top_collect(knn_item_t * knns, size_t k, long long idx, real similarity);
  void top_sort(knn_item_t * knns, size_t k);
};

#endif
