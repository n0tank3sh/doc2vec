#ifndef COMMON_DEFINE_H
#define COMMON_DEFINE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <limits>

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40
#define MAX_DOC2VEC_KNN 2000

const int negtive_sample_table_size = 1e8;

typedef float real;

#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define MIN(a,b) ( ((a)>(b)) ? (b):(a) )

#endif
