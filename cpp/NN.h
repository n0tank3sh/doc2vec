#ifndef NN_H
#define NN_H

#include "common_define.h"

#include <cstdio>

class NN
{
public:
  NN() : m_hs(false), m_negative(false), m_vocab_size(0), m_corpus_size(0), m_dim(0) { }
  NN(size_t vocab_size, size_t corpus_size, size_t dim, bool hs, int negative);
  ~NN();

  void save(FILE * fout) const;
  void load(FILE * fin);
  void norm();

  size_t dim() const { return m_dim; }
  
  bool m_hs;
  int m_negative;
  real *m_syn0 = nullptr, *m_dsyn0 = nullptr, *m_syn1 = nullptr, *m_syn1neg = nullptr;
  size_t m_vocab_size, m_corpus_size;
  //no need to flush to disk
  real * m_syn0norm = nullptr, * m_dsyn0norm = nullptr;

 private:
  size_t m_dim;
};

#endif
