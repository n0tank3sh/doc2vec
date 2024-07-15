#ifndef _DOC2VEC_TRAINMODELTHREAD_H_
#define _DOC2VEC_TRAINMODELTHREAD_H_

#include <common_define.h>

#include <vector>
#include <ctime>
#include <memory>

namespace doc2vec {
  class Model;
  class TaggedBrownCorpus;
  class TaggedDocument;

  class TrainModelThread {
    friend class Model;
  public:
    TrainModelThread(long long id, Model * doc2vec,
		     std::unique_ptr<TaggedBrownCorpus> sub_corpus, bool infer = false);

    void train();

  private:
    void updateLR();
    void buildDocument(TaggedDocument & doc, int skip = -1);
    void trainSampleCbow(long long central, long long context_start, long long context_end);
    void trainPairSg(long long central_word, real * context);
    void trainSampleSg(long long central, long long context_start, long long context_end);
    void trainDocument();
    bool down_sample(long long cn);
    long long negative_sample();
    real doc_likelihood();
    real context_likelihood(long long sentence_position);
    real likelihoodPair(long long central, real * context_vector);

    long long m_id;
    Model * m_doc2vec;
    std::unique_ptr<TaggedBrownCorpus> m_corpus;
    bool m_infer;

    clock_t m_start;
    unsigned long long m_next_random;

    std::vector<long long> m_sen;
    std::vector<long long> m_sen_nosample;
    real * m_doc_vector;
    long long m_word_count = 0;
    long long m_last_word_count = 0;
    std::unique_ptr<real[]> m_neu1;
    std::unique_ptr<real[]> m_neu1e;
  };
};

#endif
