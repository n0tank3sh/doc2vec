#include "NN.h"

#include <cmath>

NN::NN(size_t vocab_size, size_t corpus_size, size_t dim, bool hs, int negative):
  m_hs(hs), m_negative(negative),
  m_vocab_size(vocab_size), m_corpus_size(corpus_size), m_dim(dim)
{
  unsigned long long next_random = 1;
  
  posix_memalign((void **)&m_syn0, 128, (long long)m_vocab_size * m_dim * sizeof(real));
  if (m_syn0 == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}

  posix_memalign((void **)&m_dsyn0, 128, (long long)m_corpus_size * m_dim * sizeof(real));
  if (m_dsyn0 == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
  
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
    posix_memalign((void **)&m_syn1, 128, (long long)m_vocab_size * m_dim * sizeof(real));
    if (m_syn1 == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
    for (size_t a = 0; a < m_vocab_size; a++) {
      for (size_t b = 0; b < m_dim; b++) {
	m_syn1[a * m_dim + b] = 0;
      }
    }    
  }
  if (m_negative) {
    posix_memalign((void **)&m_syn1neg, 128, (long long)m_vocab_size * m_dim * sizeof(real));
    if (m_syn1neg == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
    for (size_t a = 0; a < m_vocab_size; a++) {
      for (size_t b = 0; b < m_dim; b++) {
	m_syn1neg[a * m_dim + b] = 0;
      }
    }
  }
}

NN::~NN()
{
  if(m_syn0) free(m_syn0);
  if(m_dsyn0) free(m_dsyn0);
  if(m_syn1) free(m_syn1);
  if(m_syn1neg) free(m_syn1neg);
  if(m_syn0norm) free(m_syn0norm);
  if(m_dsyn0norm) free(m_dsyn0norm);
}

void NN::save(FILE * fout) const
{
  int hs = m_hs;
  
  fwrite(&hs, sizeof(int), 1, fout);
  fwrite(&m_negative, sizeof(int), 1, fout);
  fwrite(&m_vocab_size, sizeof(size_t), 1, fout);
  fwrite(&m_corpus_size, sizeof(size_t), 1, fout);
  fwrite(&m_dim, sizeof(size_t), 1, fout);
  fwrite(m_syn0, sizeof(real), m_vocab_size * m_dim, fout);
  fwrite(m_dsyn0, sizeof(real), m_corpus_size * m_dim, fout);
  if (m_hs) fwrite(m_syn1, sizeof(real), m_vocab_size * m_dim, fout);
  if (m_negative) fwrite(m_syn1neg, sizeof(real), m_vocab_size * m_dim, fout);
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
  
  posix_memalign((void **)&m_syn0, 128, (long long)m_vocab_size * m_dim * sizeof(real));
  if (m_syn0 == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
  fread(m_syn0, sizeof(real), m_vocab_size * m_dim, fin);

  posix_memalign((void **)&m_dsyn0, 128, (long long)m_corpus_size * m_dim * sizeof(real));
  if (m_dsyn0 == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
  fread(m_dsyn0, sizeof(real), m_corpus_size * m_dim, fin);

  if (m_hs) {
    posix_memalign((void **)&m_syn1, 128, (long long)m_vocab_size * m_dim * sizeof(real));
    if (m_syn1 == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
    fread(m_syn1, sizeof(real), m_vocab_size * m_dim, fin);
  }

  if (m_negative) {
    posix_memalign((void **)&m_syn1neg, 128, (long long)m_vocab_size * m_dim * sizeof(real));
    if (m_syn1neg == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
    fread(m_syn1neg, sizeof(real), m_vocab_size * m_dim, fin);
  }
}

void NN::norm()
{
  posix_memalign((void **)&m_syn0norm, 128, (long long)m_vocab_size * m_dim * sizeof(real));
  if (m_syn0norm == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
  posix_memalign((void **)&m_dsyn0norm, 128, (long long)m_corpus_size * m_dim * sizeof(real));
  if (m_dsyn0norm == NULL) {fprintf(stderr, "Memory allocation failed\n"); exit(1);}
  
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
