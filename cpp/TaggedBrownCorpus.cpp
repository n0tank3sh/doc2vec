#include "TaggedBrownCorpus.h"
#include "Vocabulary.h"
#include "NN.h"
#include "Doc2Vec.h"

#include <set>
#include <map>
#include <vector>
#include <cmath>

//=======================TaggedBrownCorpus=======================
TaggedBrownCorpus::TaggedBrownCorpus(const char * train_file, long long seek, long long limit_doc):
  m_seek(seek), m_doc_num(0), m_limit_doc(limit_doc)
{
  m_fin = fopen(train_file, "rb");
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
UnWeightedDocument::UnWeightedDocument() : m_words_idx(NULL), m_word_num(0) {}

UnWeightedDocument::UnWeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc):
  m_words_idx(NULL), m_word_num(0)
{
  long long word_idx;
  std::set<long long> dict;
  std::vector<long long> words_idx;
  for(size_t a = 0; a < doc->m_words.size(); a++)
  {
    auto & word = doc->m_words[a];
    word_idx = doc2vec->m_word_vocab->searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    if(dict.find(word_idx) == dict.end()){
      dict.insert(word_idx);
      words_idx.push_back(word_idx);
    }
  }
  m_word_num = words_idx.size();
  if(m_word_num == 0) return;
  m_words_idx = new long long[m_word_num];
  for(size_t a = 0; a < m_word_num; a++) m_words_idx[a] = words_idx[a];
}

UnWeightedDocument::~UnWeightedDocument()
{
  delete [] m_words_idx;
}

void UnWeightedDocument::save(FILE * fout) const
{
  fwrite(&m_word_num, sizeof(int), 1, fout);
  if(m_word_num > 0) fwrite(m_words_idx, sizeof(long long), m_word_num, fout);
}
void UnWeightedDocument::load(FILE * fin)
{
  fread(&m_word_num, sizeof(int), 1, fin);
  if(m_word_num > 0)
  {
    m_words_idx = new long long[m_word_num];
    fread(m_words_idx, sizeof(long long), m_word_num, fin);
  }
  else m_words_idx = NULL;
}

// //////////////WeightedDocument/////////////////////////////
WeightedDocument::WeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc):
  UnWeightedDocument(doc2vec, doc), m_words_wei(NULL)
{
  m_words_wei = NULL;
  
  long long word_idx;
  real sim, * doc_vector = NULL, * infer_vector = NULL;
  real sum = 0;
  std::map<long long, real> scores;
  doc_vector = NULL;
  infer_vector = NULL;  
  posix_memalign((void **)&doc_vector, 128, doc2vec->m_nn->m_dim * sizeof(real));
  posix_memalign((void **)&infer_vector, 128, doc2vec->m_nn->m_dim * sizeof(real));
  doc2vec->infer_doc(*doc, doc_vector);
  for(size_t a = 0; a < doc->m_words.size(); a++)
  {
    auto & word = doc->m_words[a];
    word_idx = doc2vec->m_word_vocab->searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    doc2vec->infer_doc(*doc, infer_vector, a);
    sim = doc2vec->similarity(doc_vector, infer_vector);
    scores[word_idx] = pow(1.0 - sim, 1.5);
  }
  free(doc_vector);
  free(infer_vector);
  if(m_word_num <= 0) return;
  m_words_wei = new real[m_word_num];
  for(size_t a = 0; a < m_word_num; a++) m_words_wei[a] = scores[m_words_idx[a]];
  for(size_t a = 0; a < m_word_num; a++) sum +=  m_words_wei[a];
  for(size_t a = 0; a < m_word_num; a++) m_words_wei[a] /= sum;
}

WeightedDocument::~WeightedDocument()
{
  delete [] m_words_wei;
}
