#include "TaggedBrownCorpus.h"
#include "Vocabulary.h"
#include "NN.h"
#include "Doc2Vec.h"

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cmath>

//=======================TaggedBrownCorpus=======================
TaggedBrownCorpus::TaggedBrownCorpus(const std::string & train_file, long long seek, long long limit_doc):
  m_seek(seek), m_doc_num(0), m_limit_doc(limit_doc)
{
  m_fin = fopen(train_file.c_str(), "rb");
  if (m_fin == NULL)
  {
    fprintf(stderr, "ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(m_fin, m_seek, SEEK_SET);
}

TaggedBrownCorpus::~TaggedBrownCorpus()
{
  if(m_fin != NULL) fclose(m_fin);
  m_fin = NULL;
}

void TaggedBrownCorpus::rewind()
{
  fseek(m_fin, m_seek, SEEK_SET);
  m_doc_num = 0;
}

TaggedDocument * TaggedBrownCorpus::next()
{
  if(feof(m_fin) || (m_limit_doc >= 0 && m_doc_num >= m_limit_doc))
  {
    return NULL;
  }
  m_doc.clear();
  readWord(m_doc.m_tag);
  while ( !feof(m_fin) )
  {   
    std::string word;
    auto r = readWord(word);
    m_doc.m_words.push_back(word);
    
    if (r == -1) break;
  }
  m_doc_num++;
  return &m_doc;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// paading </s> to the EOL
//return 0 : word, return -1: EOL
int TaggedBrownCorpus::readWord(std::string & word)
{
  word.clear();
  int a = 0, ch;  
  while ( 1 )
  {
    ch = fgetc(m_fin);
    if (feof(m_fin)) {
      if (a > 0) return 0;
      return -1;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
    {
      if (a > 0)
      {
        if (ch == '\n') ungetc(ch, m_fin);
        break;
      }
      if (ch == '\n')
      {
        word = "</s>";
        return -1;
      } else continue;
    }
    word += ch;
    a++;
  }
  return 0;
}

// //////////////UnWeightedDocument/////////////////////////////
UnWeightedDocument::UnWeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc)
{
  std::unordered_set<long long> dict;
  for(size_t a = 0; a < doc->m_words.size(); a++)
  {
    auto & word = doc->m_words[a];
    auto word_idx = doc2vec->wvocab().searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    if (!dict.count(word_idx)) {
      dict.insert(word_idx);
      m_words_idx.push_back(word_idx);
    }
  }
}

void UnWeightedDocument::save(FILE * fout) const
{
  unsigned int word_num = m_words_idx.size();
  fwrite(&word_num, sizeof(unsigned int), 1, fout);
  for (auto & idx : m_words_idx) fwrite(&idx, sizeof(long long), 1, fout);
}

void UnWeightedDocument::load(FILE * fin)
{
  m_words_idx.clear();
  
  unsigned int word_num;
  fread(&word_num, sizeof(unsigned int), 1, fin);
  for (size_t i = 0; i < word_num; i++) {
    long long idx;
    fread(&idx, sizeof(long long), 1, fin);
    m_words_idx.push_back(idx);
  }
}

// //////////////WeightedDocument/////////////////////////////
WeightedDocument::WeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc):
  UnWeightedDocument(doc2vec, doc)
{
  long long word_idx;
  real * doc_vector = nullptr, * infer_vector = nullptr;
  std::unordered_map<long long, real> scores;
  posix_memalign((void **)&doc_vector, 128, doc2vec->nn().dim() * sizeof(real));
  posix_memalign((void **)&infer_vector, 128, doc2vec->nn().dim() * sizeof(real));
  doc2vec->infer_doc(*doc, doc_vector);
  for(size_t a = 0; a < doc->m_words.size(); a++)
  {
    auto & word = doc->m_words[a];
    word_idx = doc2vec->wvocab().searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    doc2vec->infer_doc(*doc, infer_vector, a);
    real sim = doc2vec->similarity(doc_vector, infer_vector);
    scores[word_idx] = pow(1.0 - sim, 1.5);
  }
  free(doc_vector);
  free(infer_vector);

  real sum = 0;
  for (auto & idx : m_words_idx) {
    auto w = scores[idx];
    m_words_wei.push_back(w);
    sum += w;
  }

  for (auto & w : m_words_wei) w /= sum;
}
