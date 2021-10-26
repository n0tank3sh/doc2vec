#include "NN.h"

#include <cmath>
#include <algorithm>

NN::NN(size_t vocab_size, size_t corpus_size, size_t dim, bool hs, int negative):
  m_hs(hs), m_negative(negative),
  m_vocab_size(vocab_size), m_corpus_size(corpus_size), m_dim(dim)
{
  unsigned long long next_random = 1;
  
  m_syn0 = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
  m_dsyn0 = std::unique_ptr<real[]>(new real[m_corpus_size * m_dim]);
  
  for (size_t a = 0; a < m_vocab_size; a++) {
    for (size_t b = 0; b < m_dim; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      m_syn0[a * m_dim + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / m_dim;
    }
  }
  
  for (size_t a = 0; a < m_corpus_size; a++) {
    for (size_t b = 0; b < m_dim; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      m_dsyn0[a * m_dim + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / m_dim;
    }
  }

  if (m_hs) {
    m_syn1 = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
    std::fill(m_syn1.get(), m_syn1.get() + m_vocab_size * m_dim, 0);    
  }
  if (m_negative) {
    m_syn1neg = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
    std::fill(m_syn1neg.get(), m_syn1neg.get() + m_vocab_size * m_dim, 0);
  }
}

void NN::save(FILE * fout) const
{
  int hs = m_hs;
  
  fwrite(&hs, sizeof(int), 1, fout);
  fwrite(&m_negative, sizeof(int), 1, fout);
  fwrite(&m_vocab_size, sizeof(size_t), 1, fout);
  fwrite(&m_corpus_size, sizeof(size_t), 1, fout);
  fwrite(&m_dim, sizeof(size_t), 1, fout);
  fwrite(m_syn0.get(), sizeof(real), m_vocab_size * m_dim, fout);
  fwrite(m_dsyn0.get(), sizeof(real), m_corpus_size * m_dim, fout);
  if (m_hs) fwrite(m_syn1.get(), sizeof(real), m_vocab_size * m_dim, fout);
  if (m_negative) fwrite(m_syn1neg.get(), sizeof(real), m_vocab_size * m_dim, fout);
}

void NN::load(FILE * fin)
{
  int hs;
  fread(&hs, sizeof(int), 1, fin);
  fread(&m_negative, sizeof(int), 1, fin);
  fread(&m_vocab_size, sizeof(size_t), 1, fin);
  fread(&m_corpus_size, sizeof(size_t), 1, fin);
  fread(&m_dim, sizeof(size_t), 1, fin);

  m_hs = hs;

  m_syn0 = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
  fread(m_syn0.get(), sizeof(real), m_vocab_size * m_dim, fin);

  m_dsyn0 = std::unique_ptr<real[]>(new real[m_corpus_size * m_dim]);
  fread(m_dsyn0.get(), sizeof(real), m_corpus_size * m_dim, fin);

  if (m_hs) {
    m_syn1 = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
    fread(m_syn1.get(), sizeof(real), m_vocab_size * m_dim, fin);
  } else {
    m_syn1.reset(nullptr);
  }

  if (m_negative) {
    m_syn1neg = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
    fread(m_syn1neg.get(), sizeof(real), m_vocab_size * m_dim, fin);
  } else {
    m_syn1neg.reset(nullptr);
  }
}

void NN::norm()
{
  m_syn0norm = std::unique_ptr<real[]>(new real[m_vocab_size * m_dim]);
  m_dsyn0norm = std::unique_ptr<real[]>(new real[m_corpus_size * m_dim]);
  
  for (size_t a = 0; a < m_vocab_size; a++) {
    real len = 0;
    for (size_t b = 0; b < m_dim; b++) {
      len += m_syn0[b + a * m_dim] * m_syn0[b + a * m_dim];
    }
    len = sqrt(len);
    for (size_t b = 0; b < m_dim; b++) m_syn0norm[b + a * m_dim] = m_syn0[b + a * m_dim] / len;
  }
  for (size_t a = 0; a < m_corpus_size; a++) {
    real len = 0;
    for (size_t b = 0; b < m_dim; b++) {
      len += m_dsyn0[b + a * m_dim] * m_dsyn0[b + a * m_dim];
    }
    len = sqrt(len);
    for (size_t b = 0; b < m_dim; b++) m_dsyn0norm[b + a * m_dim] = m_dsyn0[b + a * m_dim] / len;
  }
}
