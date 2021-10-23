#ifndef VOCAB_H
#define VOCAB_H

#include "common_define.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

struct vocab_word_t
{
  vocab_word_t() : cn(0), codelen(0), point(0), code(0) { }
  vocab_word_t(const std::string & _word, size_t _cn = 1) : word(_word), cn(_cn), codelen(0), point(0), code(0) { }
  vocab_word_t(const vocab_word_t & other) : word(other.word), cn(other.cn), codelen(other.codelen) {
    point = (int *)malloc(codelen * sizeof(int));
    code = (char *)malloc(codelen * sizeof(char));
    memcpy(point, other.point, codelen * sizeof(int));
    memcpy(code, other.code, codelen * sizeof(char));
  }  
  ~vocab_word_t() {
    free(point);
    free(code);
  }
  vocab_word_t & operator=(const vocab_word_t & other) {
    if (&other != this) {
      word = other.word;
      cn = other.cn;
      codelen = other.codelen;
      
      free(point);
      free(code);

      point = (int *)malloc(codelen * sizeof(int));
      code = (char *)malloc(codelen * sizeof(char));
      memcpy(point, other.point, codelen * sizeof(int));
      memcpy(code, other.code, codelen * sizeof(char));
    }
    return *this;
  }

  std::string word; // word string
  size_t cn; // frequency of word
  char codelen; // Hoffman code length
  int *point; // Huffman tree(n leaf + n inner node, exclude root) path. (root, leaf], node index
  char *code; // Huffman code. (root, leaf], 0/1 codes
};

class Vocabulary
{
public:
  Vocabulary() : m_min_count(1), m_doctag(false) { }
  Vocabulary(const std::string & train_file, int min_count = 5, bool doctag = false);

  long long searchVocab(const std::string & word) const;
  long long getVocabSize() const { return m_vocab.size(); }
  long long getTrainWords() const { return m_train_words; }
  void save(FILE * fout) const;
  void load(FILE * fin);
  
  size_t size() const { return m_vocab.size(); }
  const std::vector<vocab_word_t> & getWords() const { return m_vocab; }

private:
  void loadFromTrainFile(const std::string & train_file);
  void addWordToVocab(const std::string & word, size_t initial_count = 1);
  void sortVocab();
  void createHuffmanTree();

 private:
  //first place is <s>, others sorted by its frequency reversely
  std::vector<vocab_word_t> m_vocab;
  //total words of corpus. ie. sum up all frequency of words(exculude <s>)
  size_t m_train_words = 0;
  //index: hash code of a word, value: vocab index of the word
  std::unordered_map<std::string, size_t> m_vocab_hash;
  int m_min_count;
  bool m_doctag;
};

#endif
