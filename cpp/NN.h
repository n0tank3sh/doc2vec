#ifndef NN_H
#define NN_H

#include "common_define.h"

#include <memory>
#include <cstdio>

class NN
{
public:
  NN() : m_hs(false), m_negative(false), m_vocab_size(0), m_corpus_size(0), m_dim(0) { }
  NN(size_t vocab_size, size_t corpus_size, size_t dim, bool hs, int negative);

  void save(FILE * fout) const;
  void load(FILE * fin);
  void norm();

  size_t dim() const { return m_dim; }
  real * get_syn0() { return m_syn0.get(); }
  real * get_dsyn0() { return m_dsyn0.get(); }
  real * get_syn1() { return m_syn1.get(); }
  real * get_syn1neg() { return m_syn1neg.get(); }
  const real * get_syn0norm() const { return m_syn0norm.get(); }
  const real * get_dsyn0norm() const { return m_dsyn0norm.get(); }
  
  bool m_hs;
  int m_negative;
  size_t m_vocab_size, m_corpus_size;

 private:
  size_t m_dim;
  std::unique_ptr<real[]> m_syn0, m_dsyn0, m_syn1, m_syn1neg;

  //no need to flush to disk
  std::unique_ptr<real[]> m_syn0norm, m_dsyn0norm;
};

#endif
