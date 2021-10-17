# Change log
	
## 2021-10-17

### Import changes from https://github.com/bnosac/doc2vec/
	
- In paragraph2vec: close opened files directly after training instead of waiting for R garbage collection to kick in
- Start fixing valgrind issues
- more valgrind changes

## 2021-10-18
	
### Change

- replace custom hash with std::unordered_map, to fix issue with more than 30M documents
	
