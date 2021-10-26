# Change log

## 2021-10-17

### Import changes from https://github.com/bnosac/doc2vec/

- In paragraph2vec: close opened files directly after training instead of waiting for R garbage collection to kick in
- Start fixing valgrind issues
- more valgrind changes

## 2021-10-18

### Change

- replace custom hash with std::unordered_map, to fix issue with more than 30M documents
- print debug info to stderr instead of stdout
- rename Vocab.{cpp,h} to Vocabulary.{cpp,h}
- refactor by adding const where suitable
- remove document and word size limits and refactor to modernize code
- check return value from output file creation
- return 1 on argument error in train.cpp

## 2021-10-19

- Use unordered_set and unordered_map instead of set and map
	

## 2021-10-21

- Instead of ftell() use ftello(), which is quaranteed to be 64 bit
	

## 2021-10-23

- Enable fast-math, which almost doubles the performance

## 2021-10-27

- Remove aligned allocations, since they have no effect on performance
