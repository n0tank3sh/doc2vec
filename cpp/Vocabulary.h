#ifndef VOCAB_H
#define VOCAB_H

#include "common_define.h"

#include <unordered_map>
#include <vector>
#include <string>

struct vocab_word_t
{
  vocab_word_t(size_t _cn = 1) : cn(_cn) { }
  ~vocab_word_t() {
    free(word);
    free(point);
    free(code);
  }

  size_t cn; //frequency of word
  int *point = 0; //Huffman tree(n leaf + n inner node, exclude root) path. (root, leaf], node index
  char *word = 0; //word string
  char *code = 0; //Huffman code. (root, leaf], 0/1 codes
  char codelen = 0; //Hoffman code length
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
