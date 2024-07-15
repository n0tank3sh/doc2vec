#include <Vocabulary.h>
#include <TaggedBrownCorpus.h>

#include <cassert>
#include <cstring>
#include <algorithm>
#include <memory>

using namespace doc2vec;

static inline bool vocabCompare(const vocab_word_t & a, const vocab_word_t & b)
{
    return a.cn > b.cn;
}

Vocabulary::Vocabulary(Input & train_file, int min_count, bool doctag)
  : m_min_count(min_count), m_doctag(doctag)
{
  if(m_doctag) m_min_count = 1;
  loadFromTrainFile(train_file);
  if(!m_doctag) createHuffmanTree();
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
long long Vocabulary::searchVocab(const std::string & word) const
{
  auto it = m_vocab_hash.find(word);
  if (it != m_vocab_hash.end()) return it->second;
  else return -1;
}

void Vocabulary::loadFromTrainFile(Input & train_file) {
  TaggedBrownCorpus corpus(train_file);
  m_vocab.clear();
  m_vocab_hash.clear();
  if(!m_doctag) addWordToVocab("</s>", 0);
  TaggedDocument * doc = NULL;
  while ((doc = corpus.next()) != NULL) {
    if(m_doctag) {  //for doc tag
      auto & word = doc->m_tag;
      m_train_words++;
      long long i = searchVocab(word);
      if (i == -1) {
	fprintf(stderr, "adding doc: %s\n", word.c_str());
	addWordToVocab(word);
      }
    } else { // for doc words
      for(size_t k = 0; k < doc->m_words.size(); k++){
        auto & word = doc->m_words[k];
        m_train_words++;
        if (!m_doctag && m_train_words % 100000 == 0)
        {
          fprintf(stderr, "%lldK%c", m_train_words / 1000, 13);
          fflush(stderr);
        }
        long long i = searchVocab(word);
        if (i == -1) addWordToVocab(word);
	else m_vocab[i].cn++;
      }
      m_train_words--;
    }
  }
  if(!m_doctag)
  {
    sortVocab();
    fprintf(stderr, "Vocab size: %lld\n", m_vocab.size());
    fprintf(stderr, "Words in train file: %lld\n", m_train_words);
  }
}

void Vocabulary::addWordToVocab(const std::string & word, size_t initial_count) 
{
  vocab_word_t w(word, initial_count);    
  m_vocab_hash[word] = m_vocab.size();
  m_vocab.push_back(w);
}

// Sorts the vocabulary by frequency using word counts, frequent->infrequent
void Vocabulary::sortVocab()
{
  assert(!m_vocab.empty());
  fprintf(stderr, "sorting\n");
  // Sort the vocabulary and keep </s> at the first position  
  std::sort(m_vocab.begin() + 1, m_vocab.end(), vocabCompare);
  //reduce words and re-hash
  m_vocab_hash.clear();
  m_train_words = 0;
  fprintf(stderr, "removing\n");
  while (m_vocab.size() > 1 && m_vocab.back().cn < m_min_count) {
    // Words occuring less than min_count times will be discarded from the vocab
    m_vocab.pop_back();
  }
  fprintf(stderr, "rehashing\n");
  for (size_t i = 0; i < m_vocab.size(); i++) {
    // Hash will be re-computed, as after the sorting it is not actual
    m_vocab_hash[m_vocab[i].word] = i;
    m_train_words += m_vocab[i].cn;
  }
  m_train_words -= m_vocab.front().cn; //exclude <s>
  m_vocab.shrink_to_fit();
}

void Vocabulary::createHuffmanTree()
{
  // Allocate memory for the binary tree construction
  long long b, i, min1i, min2i, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  std::unique_ptr<long long[]> count(new long long[m_vocab.size() * 2 + 1]);
  std::unique_ptr<long long[]> binary(new long long[m_vocab.size() * 2 + 1]);
  std::unique_ptr<long long[]> parent_node(new long long[m_vocab.size() * 2 + 1]);
  for (size_t a = 0; a < m_vocab.size(); a++) {
    m_vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    m_vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    count[a] = m_vocab[a].cn;
  }
  for (size_t a = m_vocab.size(); a < m_vocab.size() * 2; a++) count[a] = 1e15;
  long long pos1 = m_vocab.size() - 1;
  long long pos2 = m_vocab.size();
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (size_t a = 0; a < m_vocab.size() - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[m_vocab.size() + a] = count[min1i] + count[min2i];
    parent_node[min1i] = m_vocab.size() + a;
    parent_node[min2i] = m_vocab.size() + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (size_t a = 0; a < m_vocab.size(); a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == m_vocab.size() * 2 - 2) break;
    }
    m_vocab[a].codelen = i;
    m_vocab[a].point[0] = m_vocab.size() - 2;
    for (b = 0; b < i; b++) {
      m_vocab[a].code[i - b - 1] = code[b];
      m_vocab[a].point[i - b] = point[b] - m_vocab.size();
    }
  }
}

void Vocabulary::save(FILE * fout) const
{
  long long dummy = 0, size = m_vocab.size();
  
  fwrite(&size, sizeof(long long), 1, fout);
  fwrite(&m_train_words, sizeof(long long), 1, fout);
  fwrite(&dummy, sizeof(long long), 1, fout);
  fwrite(&m_min_count, sizeof(int), 1, fout);
  fwrite(&m_doctag, sizeof(bool), 1, fout);
  for (auto & w : m_vocab) {
    unsigned int wordlen = w.word.size();
    fwrite(&wordlen, sizeof(unsigned int), 1, fout);
    fwrite(w.word.data(), sizeof(char), wordlen, fout);
    fwrite(&(w.cn), sizeof(size_t), 1, fout);
    if(!m_doctag)
    {
      fwrite(&(w.codelen), sizeof(char), 1, fout);
      fwrite(w.point, sizeof(int), w.codelen, fout);
      fwrite(w.code, sizeof(char), w.codelen, fout);
    }
  }
}

void Vocabulary::load(FILE * fin)
{
  size_t size, dummy;
  fread(&size, sizeof(size_t), 1, fin);
  fread(&m_train_words, sizeof(size_t), 1, fin);
  fread(&dummy, sizeof(size_t), 1, fin);
  fread(&m_min_count, sizeof(int), 1, fin);
  fread(&m_doctag, sizeof(bool), 1, fin);

  m_vocab.clear();
  m_vocab_hash.clear();
  
  for (size_t a = 0; a < size; a++) {    
    unsigned int wordlen;
    fread(&wordlen, sizeof(int), 1, fin);

    char tmp[wordlen + 1];
    fread(tmp, sizeof(char), wordlen, fin);
    tmp[wordlen] = 0;       

    size_t cn;
    fread(&cn, sizeof(size_t), 1, fin);
    
    vocab_word_t w(tmp, cn);

    if (!m_doctag) {
      fread(&(w.codelen), sizeof(char), 1, fin);
      w.point = (int *)calloc(w.codelen, sizeof(int));
      fread(w.point, sizeof(int), w.codelen, fin);
      w.code = (char *)calloc(w.codelen, sizeof(char));
      fread(w.code, sizeof(char), w.codelen, fin);      
    }
    m_vocab_hash[w.word] = m_vocab.size();
    m_vocab.push_back(w);
  }
}
