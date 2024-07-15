#include <TaggedBrownCorpus.h>
#include <Vocabulary.h>
#include <NN.h>
#include <Model.h>

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cmath>

using namespace doc2vec;

TaggedBrownCorpus::TaggedBrownCorpus(Input & train_file, long long seek, long long limit_doc)
  : m_seek(seek), m_doc_num(0), m_limit_doc(limit_doc), m_train_file(train_file.copy())
{
  m_train_file->seek(m_seek);
}

void TaggedBrownCorpus::rewind() {
  m_train_file->seek(m_seek);
  m_doc_num = 0;
}

TaggedDocument * TaggedBrownCorpus::next()
{
  if (m_train_file->eof() || (m_limit_doc >= 0 && m_doc_num >= m_limit_doc)) {
    return NULL;
  }
  m_doc.clear();
  auto line = m_train_file->get_line();
  if (line.empty()) return NULL;
  m_doc.m_tag = line[0];
  for (size_t i = 1; i < line.size(); i++) {
    m_doc.m_words.push_back(line[i]);
  }
  m_doc_num++;
  return &m_doc;
}

UnWeightedDocument::UnWeightedDocument(Model * doc2vec, TaggedDocument * doc)
{
  std::unordered_set<long long> dict;
  for (size_t a = 0; a < doc->m_words.size(); a++) {
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

WeightedDocument::WeightedDocument(Model * doc2vec, TaggedDocument * doc)
  : UnWeightedDocument(doc2vec, doc)
{
  long long word_idx;
  std::unordered_map<long long, real> scores;
  std::unique_ptr<real[]> doc_vector(new real[doc2vec->nn().dim()]);
  std::unique_ptr<real[]> infer_vector(new real[doc2vec->nn().dim()]);
  doc2vec->infer_doc(*doc, doc_vector.get());
  for (size_t a = 0; a < doc->m_words.size(); a++) {
    auto & word = doc->m_words[a];
    word_idx = doc2vec->wvocab().searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    doc2vec->infer_doc(*doc, infer_vector.get(), a);
    real sim = doc2vec->similarity(doc_vector.get(), infer_vector.get());
    scores[word_idx] = pow(1.0 - sim, 1.5);
  }

  real sum = 0;
  for (auto & idx : m_words_idx) {
    auto w = scores[idx];
    m_words_wei.push_back(w);
    sum += w;
  }

  for (auto & w : m_words_wei) w /= sum;
}
