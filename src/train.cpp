#include <common_define.h>
#include <Model.h>
#include <Input.h>

#include <cstring>

using namespace doc2vec;

// setup parameters
std::string train_file, output_file;
bool cbow = true;
int window = 5, min_count = 1, num_threads = 4;
bool hs = 1;
int negative = 0;
long long dim = 100, iter = 50;
real alpha = 0.025, sample = 1e-3;

static int ArgPos(char *str, int argc, char **argv);
static void usage();
static int get_optarg(int argc, char **argv);

int ArgPos(char *str, int argc, char **argv)
{
  int a;
  for (a = 1; a < argc; a++) {
    if (strcmp(str, argv[a]) == 0) {
      if (a == argc - 1) {
	fprintf(stderr, "Argument missing for %s\n", str);
	exit(1);
      }
      return a;
    }
  }
  return -1;
}

//usage of main
void usage()
{
  fprintf(stderr, "DOCUMENT/WORD VECTOR estimation toolkit\n\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "Parameters for training:\n");
  fprintf(stderr, "\t-train <file>\n");
  fprintf(stderr, "\t\tUse text data from <file> to train the model\n");
  fprintf(stderr, "\t-output <file>\n");
  fprintf(stderr, "\t\tUse <file> to save the resulting model\n");
  fprintf(stderr, "\t-dim <int>\n");
  fprintf(stderr, "\t\tSet dimention of document/word vectors; default is 100\n");
  fprintf(stderr, "\t-window <int>\n");
  fprintf(stderr, "\t\tSet max skip length between words; default is 5\n");
  fprintf(stderr, "\t-sample <float>\n");
  fprintf(stderr, "\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
  fprintf(stderr, "\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
  fprintf(stderr, "\t-threads <int>\n");
  fprintf(stderr, "\t\tUse <int> threads (default 12)\n");
  fprintf(stderr, "\t-iter <int>\n");
  fprintf(stderr, "\t\tRun more training iterations (default 5)\n");
  fprintf(stderr, "\t-min-count <int>\n");
  fprintf(stderr, "\t\tThis will discard words that appear less than <int> times; default is 5\n");
  fprintf(stderr, "\t-alpha <float>\n");
  fprintf(stderr, "\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
  fprintf(stderr, "\t-cbow <int>\n");
  fprintf(stderr, "\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
  fprintf(stderr, "\t-hs <int>\n");
  fprintf(stderr, "\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
  fprintf(stderr, "\t-negative <int>\n");
  fprintf(stderr, "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
}

//get arguments from command line
int get_optarg(int argc, char **argv)
{
  int i;
  if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) train_file = argv[i + 1];
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]) ? true : false;
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) output_file = argv[i + 1];
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  return output_file.empty() ? -1 : 0;
}

int main(int argc, char **argv)
{ 
  if (argc == 1 || get_optarg(argc, argv) < 0)
  {
    usage();
    return 1;
  }
  FILE * fout = fopen(output_file.c_str(), "wb");
  if (!fout) {
    fprintf(stderr, "Unable to open file %s\n", output_file.c_str());
    return 1;
  }

  FileInput input(train_file);
  
  Model doc2vec;
  doc2vec.train(input, dim, cbow, hs, negative, iter, window, alpha, sample, min_count, num_threads);
  fprintf(stderr, "\nWrite model to %s\n", output_file.c_str());
  doc2vec.save(fout);
  fclose(fout);
  return 0;
}
