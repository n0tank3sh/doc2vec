GCC=g++
MALLOC=/usr/include/malloc
INCLUDEDIR=-I$(MALLOC)
LIBDIR=-lm -pthread
CPPFLAGS = -O3 -std=c++1z -Wall -march=native -funroll-loops -Wno-unused-result
TARGET = train
OBJS = train.o Doc2Vec.o TrainModelThread.o Vocabulary.o NN.o TaggedBrownCorpus.o WMD.o

all: ${TARGET}

clean:
	rm -f *.o $(TARGET)

${TARGET} : $(OBJS)
	$(GCC) $(CPPFLAGS) $^ $(LIBDIR) -o $@

%.o	: %.cpp
	$(GCC) $(CPPFLAGS) -c $< -o $@ $(INCLUDEDIR)
