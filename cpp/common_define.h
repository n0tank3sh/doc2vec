#ifndef COMMON_DEFINE_H
#define COMMON_DEFINE_H

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40
#define MAX_DOC2VEC_KNN 2000

const int negative_sample_table_size = 1e8;

typedef float real;

static inline real MAX(real a, real b) { return a > b ? a : b; }
static inline real MIN(real a, real b) { return a > b ? b : a; }

static inline int MAX(int a, int b) { return a > b ? a : b; }
static inline int MIN(int a, int b) { return a > b ? b : a; }

static inline long long MAX(long long a, long long b) { return a > b ? a : b; }
static inline long long MIN(long long a, long long b) { return a > b ? b : a; }

#endif
