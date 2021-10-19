#ifndef DOC2VEC_H
#define DOC2VEC_H
#include "common_define.h"

#include <vector>
#include <string>
#include <memory>

class TrainModelThread;
class NN;
class Vocabulary;
class WMD;
class TaggedBrownCorpus;
class TaggedDocument;
struct knn_item_t;

class Doc2Vec
{
friend class TrainModelThread;
friend class WMD;
friend class UnWeightedDocument;
friend class WeightedDocument;
public:
  Doc2Vec();
  ~Doc2Vec();

  void train(const std::string & train_file,
    size_t dim, bool cbow, int hs, int negtive,
    int iter, int window,
    real alpha, real sample,
    int min_count, int threads);

  size_t dim() const;
  WMD * wmd() { return m_wmd.get(); }
  Vocabulary * wvocab() { return m_word_vocab.get(); }
  Vocabulary * dvocab() { return m_doc_vocab.get(); }
  NN * nn() { return m_nn.get(); };

  real doc_likelihood(TaggedDocument & doc, int skip = -1);
  real context_likelihood(TaggedDocument & doc, int sentence_position);
  void infer_doc(TaggedDocument & doc, real * vec, int skip = -1);
  bool word_knn_words(const std::string & search, knn_item_t * knns, int k);
  bool doc_knn_docs(const std::string & search, knn_item_t * knns, int k);
  bool word_knn_docs(const std::string & search, knn_item_t * knns, int k);

  void sent_knn_words(TaggedDocument & doc, knn_item_t * knns, int k, real * infer_vector);
  void sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, int k, real * infer_vector);

  void sent_knn_words(TaggedDocument & doc, knn_item_t * knns, int k);
  void sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, int k);
  
  real similarity(real * src, real * target);
  real distance(real * src, real * target);

  void save(FILE * fout) const;
  void load(FILE * fin);

 private:
  void initExpTable();
  void initNegTable();
  void initTrainModelThreads(const std::string & train_file, int threads, int iter);
  bool obj_knn_objs(const std::string & search, real* src,
    bool search_is_word, bool target_is_word,
    knn_item_t * knns, int k);

  std::unique_ptr<Vocabulary> m_word_vocab;
  std::unique_ptr<Vocabulary> m_doc_vocab;
  std::unique_ptr<NN> m_nn;
  std::unique_ptr<WMD> m_wmd;
  
  bool m_cbow = true;
  int m_hs;
  int m_negtive;
  int m_window;
  real m_start_alpha; //fix lr
  real m_sample;
  int m_iter;

  //no need to flush to disk
  std::unique_ptr<TaggedBrownCorpus> m_brown_corpus;
  real m_alpha; //working lr
  long long m_word_count_actual;
  real * m_expTable = nullptr;
  int * m_negtive_sample_table = nullptr;
  std::vector<TrainModelThread *> m_trainModelThreads;
};

struct knn_item_t
{
  std::string word;
  long long idx;
  real similarity;
};
void top_init(knn_item_t * knns, int k);
void top_collect(knn_item_t * knns, int k, long long idx, real similarity);
void top_sort(knn_item_t * knns, int k);
#endif
