#ifndef VOCAB_H
#define VOCAB_H

#include "common_define.h"

#include <unordered_map>
#include <string>

struct vocab_word_t
{
  size_t cn; //frequency of word
  int *point; //Huffman tree(n leaf + n inner node, exclude root) path. (root, leaf], node index
  char *word; //word string
  char *code; //Huffman code. (root, leaf], 0/1 codes
  char codelen; //Hoffman code length
};

class Vocabulary
{
public:
  Vocabulary() : m_min_count(1), m_doctag(false) { }
  Vocabulary(const std::string & train_file, int min_count = 5, bool doctag = false);
  ~Vocabulary();

  long long searchVocab(const std::string & word) const;
  long long getVocabSize() const { return m_vocab_size; }
  long long getTrainWords() const { return m_train_words; }
  void save(FILE * fout) const;
  void load(FILE * fin);

private:
  void loadFromTrainFile(const std::string & train_file);
  long long addWordToVocab(const std::string & word);
  void sortVocab();
  void createHuffmanTree();

public:
  //first place is <s>, others sorted by its frequency reversely
  struct vocab_word_t* m_vocab = nullptr;
  //size of vocab including <s>
  size_t m_vocab_size = 0;
  //total words of corpus. ie. sum up all frequency of words(exculude <s>)
  size_t m_train_words = 0;
  
private:
  size_t m_vocab_capacity = 1000;
  //index: hash code of a word, value: vocab index of the word
  std::unordered_map<std::string, size_t> m_vocab_hash;
  int m_min_count;
  bool m_doctag;
};

#endif
