#ifndef _DOC2VEC_INPUT_H_
#define _DOC2VEC_INPUT_H_

#include <string>
#include <memory>
#include <vector>

namespace doc2vec {
  class Input {
  public:
    Input() { }
    virtual ~Input() { }

    virtual std::unique_ptr<Input> copy() = 0;
    virtual long long tell() = 0;
    virtual bool eof() = 0;
    virtual void seek(long long pos) = 0;
    virtual std::vector<std::string> get_line() = 0;
  };

  class FileInput : public Input {
  public:
  FileInput(std::string filename) : filename_(std::move(filename)) {
      m_fin = fopen(filename_.c_str(), "rb");
      if (m_fin == NULL) {
	fprintf(stderr, "ERROR: training data file not found!\n");
	exit(1);
      }
    }
    ~FileInput() {
      if (m_fin) fclose(m_fin);
    }
    std::unique_ptr<Input> copy() { return std::make_unique<FileInput>(filename_); }    
    bool eof() override { return feof(m_fin); }
    long long tell() override { return ftello(m_fin); }
    void seek(long long pos) override { fseek(m_fin, pos, SEEK_SET); }

    // Reads a single word from a file, assuming space + tab + EOL to be word boundaries
    // paading </s> to the EOL
    // return 0 : word, return -1: EOL
    int readWord(std::string & word) {
      word.clear();
      int a = 0, ch;  
      while ( 1 ) {
	ch = fgetc(m_fin);
	if (eof()) {
	  if (a > 0) return 0;
	  return -1;
	}
	if (ch == 13) continue;
	if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
	  if (a > 0) {
	    if (ch == '\n') ungetc(ch, m_fin);
	    break;
	  }
	  if (ch == '\n') {
	    word = "</s>";
	    return -1;
	  } else continue;
	}
	word += ch;
	a++;
      }
      return 0;
    }

    std::vector<std::string> get_line() override {
      std::string word;
      readWord(word);
      std::vector<std::string> line { word };
      while ( !eof() ) {   
	auto r = readWord(word);
	line.push_back(word);
        if (r == -1) break;
      }
      return line;
    }
    
  private:
    std::string filename_;
    FILE * m_fin;
  };

  class MemoryInput : public Input {
  public:
    MemoryInput(size_t size, const char * data) : size_(size), data_(data) { }

    std::unique_ptr<Input> copy() { return std::make_unique<MemoryInput>(size_, data_); }
    bool eof() override { return pos_ >= size_; }
    long long tell() override { return pos_; }
    void seek(long long pos) override { pos_ = pos; }

    int readWord(std::string & word) {
      word.clear();
      int a = 0, ch;  
      while ( 1 ) {
	if (pos_ < size_) ch = data_[pos_++];
	else {
	  if (a > 0) return 0;
	  return -1;
	}
	if (ch == 13) continue;
	if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
	  if (a > 0) {
	    if (ch == '\n') pos_--;
	    break;
	  }
	  if (ch == '\n') {
	    word = "</s>";
	    return -1;
	  } else continue;
	}
	word += ch;
	a++;
      }
      return 0;
    }

    std::vector<std::string> get_line() override {
      std::string word;
      readWord(word);
      std::vector<std::string> line { word };
      while ( pos_ < size_ ) {   
	auto r = readWord(word);
	line.push_back(word);
        if (r == -1) break;
      }
      return line;
    }

  private:
    size_t size_;
    const char * data_;
    size_t pos_ = 0;
  };
};


#endif
