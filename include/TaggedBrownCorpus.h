#ifndef _DOC2VEC_TAGGEDBROWNCORPUS_H_
#define _DOC2VEC_TAGGEDBROWNCORPUS_H_

#include "Input.h"
#include "common_define.h"

#include <string>
#include <vector>

namespace doc2vec {
  class Model;

  class TaggedDocument {
  public:
    TaggedDocument() { }
    TaggedDocument(const std::vector<std::string> & words) : m_words(words) { }
    
    void clear() {
      m_tag.clear();
      m_words.clear();
    }
    void addWord(const std::string & word) { m_words.push_back(word); }
  
    std::string m_tag;
    std::vector<std::string> m_words;
  };
  
  class TaggedBrownCorpus {
  public:
    TaggedBrownCorpus(Input & train_file, long long seek = 0, long long limit_doc = -1);

    TaggedDocument * next();
    void rewind();
    long long tell() { return m_train_file->tell(); }
    long long getDocNum() const { return m_doc_num; }

  private:
    TaggedDocument m_doc;
    long long m_seek;
    long long m_doc_num;
    long long m_limit_doc;
    std::unique_ptr<Input> m_train_file;
  };

  class UnWeightedDocument {
  public:
    UnWeightedDocument() { }
    UnWeightedDocument(Model * doc2vec, TaggedDocument * doc);

    void save(FILE * fout) const;
    void load(FILE * fin);

    std::vector<long long> m_words_idx;
  };

  class WeightedDocument : public UnWeightedDocument {
  public:
    WeightedDocument(Model * doc2vec, TaggedDocument * doc);
    
    std::vector<real> m_words_wei;
  };
};

#endif
