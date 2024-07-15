#include <Model.h>
#include <TrainModelThread.h>
#include <Input.h>

#include <cmath>

namespace doc2vec {
  static void heap_adjust(knn_item_t * knns, int s, int m) {
    real similarity = knns[s].similarity;
    long long idx = knns[s].idx;
    for(int j = 2 * s + 1; j < m; j = 2 * j + 1) {
      if(j < m - 1 && knns[j].similarity > knns[j + 1].similarity) j++;
      if(similarity < knns[j].similarity) break;
      knns[s].similarity = knns[j].similarity;
      knns[s].idx = knns[j].idx;
      s = j;
    }
    knns[s].similarity = similarity;
    knns[s].idx = idx;
  }

  void top_init(knn_item_t * knns, size_t k) {
    for (int i = k / 2 - 1; i >= 0; i--) {
      heap_adjust(knns, i, k);
    }
  }

  void top_collect(knn_item_t * knns, size_t k, long long idx, real similarity) {
    if (similarity <= knns[0].similarity) return;
    knns[0].similarity = similarity;
    knns[0].idx = idx;
    heap_adjust(knns, 0, k);
  }

  void top_sort(knn_item_t * knns, size_t k) {
    real similarity;
    long long idx;
    for (int i = k - 1; i > 0; i--) {
      similarity = knns[0].similarity;
      idx = knns[0].idx;
      knns[0].similarity = knns[i].similarity;
      knns[0].idx = knns[i].idx;
      knns[i].similarity = similarity;
      knns[i].idx = idx;
      heap_adjust(knns, 0, i);
    }
  }
};

using namespace doc2vec;

static void * trainModelThread(void * params);

void * trainModelThread(void * params)
{
  TrainModelThread * tparams = (TrainModelThread *)params;
  tparams->train();
  return NULL;
}

Model::Model()
{
  initExpTable();
}

void Model::initExpTable()
{
  m_expTable = std::unique_ptr<real[]>(new real[EXP_TABLE_SIZE]);

  for (int i = 0; i < EXP_TABLE_SIZE; i++)
  {
    m_expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    m_expTable[i] = m_expTable[i] / (m_expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
}

void Model::initNegTable()
{
  m_negative_sample_table = std::unique_ptr<int[]>(new int[negative_sample_table_size]);

  auto & words = m_word_vocab->getWords();
  long long train_words_pow = 0;
  real power = 0.75;
  for (size_t a = 0; a < words.size(); a++) train_words_pow += pow(words[a].cn, power);
  int i = 0;
  real d1 = pow(words[i].cn, power) / (real)train_words_pow;
  for (size_t a = 0; a < negative_sample_table_size; a++) {
    m_negative_sample_table[a] = i;
    if (a / (real)negative_sample_table_size > d1) {
      i++;
      d1 += pow(words[i].cn, power) / (real)train_words_pow;
    }
    if (i >= words.size()) i = words.size() - 1;
  }
}

void Model::train(Input & train_file,
  size_t dim, bool cbow, bool hs, int negative,
  int iter, int window,
  real alpha, real sample,
  int min_count, int threads)
{
  fprintf(stderr, "Starting training\n");
  m_cbow = cbow;
  m_hs = hs;
  m_negative = negative;
  m_window = window;
  m_start_alpha = alpha;
  m_sample = sample;
  m_iter = iter;

  m_word_vocab = std::make_unique<Vocabulary>(train_file, min_count);
  m_doc_vocab = std::make_unique<Vocabulary>(train_file, 1, true);
  m_nn = std::make_unique<NN>(m_word_vocab->size(), m_doc_vocab->size(), dim, hs, negative);
  if (m_negative > 0) initNegTable();

  fprintf(stderr, "word vocab: %d, doc vocab: %d\n", int(m_word_vocab->size()), int(m_doc_vocab->size()));
  
  m_brown_corpus = std::make_unique<TaggedBrownCorpus>(train_file);
  m_alpha = alpha;
  m_word_count_actual = 0;

  std::vector<TrainModelThread *> trainModelThreads;
  initTrainModelThreads(train_file, threads, iter, trainModelThreads);

  fprintf(stderr, "Train with %d threads\n", (int)trainModelThreads.size());
  auto pt = std::make_unique<pthread_t[]>(trainModelThreads.size());
  for (size_t a = 0; a < trainModelThreads.size(); a++) {
    pthread_create(&pt[a], NULL, trainModelThread, (void *)trainModelThreads[a]);
  }
  for (size_t a = 0; a < trainModelThreads.size(); a++) {
    pthread_join(pt[a], NULL);
    delete trainModelThreads[a];
  }

  // for(size_t i =  0; i < m_trainModelThreads.size(); i++) m_trainModelThreads[i]->m_corpus->close();
  // m_brown_corpus->close();
  
  m_nn->norm();
  m_wmd = std::make_unique<WMD>(this);
  m_wmd->train();
}

void Model::initTrainModelThreads(Input & train_file, int threads, int iter, std::vector<TrainModelThread *> & trainModelThreads)
{
  long long limit = m_doc_vocab->size() / threads;
  long long sub_size = 0;
  long long tell = 0;
  TaggedBrownCorpus brown_corpus(train_file);
  TaggedDocument * doc = NULL;
  while((doc = brown_corpus.next()) != NULL)
  {
    sub_size++;
    if(sub_size >= limit)
    {
        auto sub_c = std::make_unique<TaggedBrownCorpus>(train_file, tell, sub_size);
        auto model_thread = new TrainModelThread(trainModelThreads.size(), this, std::move(sub_c), false);
        trainModelThreads.push_back(model_thread);
        tell = brown_corpus.tell();
        sub_size = 0;
    }
  }
  if (trainModelThreads.size() < size_t(threads)) {
    auto sub_c = std::make_unique<TaggedBrownCorpus>(train_file, tell, -1);
    auto model_thread = new TrainModelThread(trainModelThreads.size(), this, std::move(sub_c), false);
    trainModelThreads.push_back(model_thread);
  }
  fprintf(stderr, "corpus size: %lld\n", m_doc_vocab->size() - 1);
}

bool Model::obj_knn_objs(const std::string & search, const real * src,
  bool search_is_word, bool target_is_word,
  knn_item_t * knns, size_t k)
{
  const Vocabulary * search_vocab = search_is_word ? m_word_vocab.get() : m_doc_vocab.get();
  const real * search_vectors = search_is_word ? m_nn->get_syn0norm() : m_nn->get_dsyn0norm();
  const real * target_vectors = target_is_word ? m_nn->get_syn0norm() : m_nn->get_dsyn0norm();
  size_t target_size = target_is_word ? m_nn->m_vocab_size : m_nn->m_corpus_size;
  const Vocabulary * target_vocab = target_is_word ? m_word_vocab.get() : m_doc_vocab.get();
  long long a;
  if (!src) {
    a = search_vocab->searchVocab(search);
    if (a < 0) {
      return false;
    }
    src = &(search_vectors[a * m_nn->dim()]);
  }
  for (size_t b = 0, c = 0; b < target_size; b++)
  {
    if (search_is_word == target_is_word && a == b) continue;
    auto target = &(target_vectors[b * m_nn->dim()]);
    if (c < k) {
      knns[c].similarity = similarity(src, target);
      knns[c].idx = b;
      c++;
      if (c == k) top_init(knns, k);
    }
    else top_collect(knns, k, b, similarity(src, target));
  }
  top_sort(knns, k);
  auto & target_words = target_vocab->getWords();
  for (size_t b = 0; b < k; b++) knns[b].word = target_words[knns[b].idx].word;
  return true;
}

bool Model::word_knn_words(const std::string & search, knn_item_t * knns, size_t k)
{
  return obj_knn_objs(search, NULL, true, true, knns, k);
}

bool Model::doc_knn_docs(const std::string & search, knn_item_t * knns, size_t k)
{
  return obj_knn_objs(search, NULL, false, false, knns, k);
}

bool Model::word_knn_docs(const std::string & search, knn_item_t * knns, size_t k)
{
  return obj_knn_objs(search, NULL, true, false, knns, k);
}

void Model::sent_knn_words(TaggedDocument & doc, knn_item_t * knns, size_t k)
{
  std::unique_ptr<real[]> infer_vector(new real[m_nn->dim()]);  
  sent_knn_words(doc, knns, k, infer_vector.get());  
}

void Model::sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, size_t k)
{
  std::unique_ptr<real[]> infer_vector(new real[m_nn->dim()]);    
  sent_knn_docs(doc, knns, k, infer_vector.get());
}

void Model::sent_knn_words(TaggedDocument & doc, knn_item_t * knns, size_t k, real * infer_vector)
{
  infer_doc(doc, infer_vector);
  obj_knn_objs("", infer_vector, false, true, knns, k);
}

void Model::sent_knn_docs(TaggedDocument & doc, knn_item_t * knns, size_t k, real * infer_vector)
{
  infer_doc(doc, infer_vector);
  obj_knn_objs("", infer_vector, false, false, knns, k);
}

real Model::similarity(const real * src, const real * target) const
{
  real dot = 0;
  for (size_t a = 0; a < m_nn->dim(); a++) dot += src[a] * target[a];
  return dot;
}

real Model::distance(const real * src, const real * target) const
{
  real dis = 0;
  for (size_t a = 0; a < m_nn->dim(); a++) dis += pow(src[a] - target[a], 2);
  return sqrt(dis);
}

void Model::infer_doc(TaggedDocument & doc, real * vec, int skip)
{
  real len = 0;
  unsigned long long next_random = 1;
  for (long long a = 0; a < m_nn->dim(); a++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    vec[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / m_nn->dim();
  }
  m_alpha = m_start_alpha;
  TrainModelThread trainThread(0, this, NULL, true);
  trainThread.m_doc_vector = vec;
  trainThread.buildDocument(doc, skip);
  for(long long a = 0; a < m_iter; a++)
  {
    trainThread.trainDocument();
    m_alpha = m_start_alpha * (1 - (a + 1.0) / m_iter);
    m_alpha = MAX(m_alpha, m_start_alpha * 0.0001);
  }
  for(long long a = 0; a < m_nn->dim(); a++) len += vec[a] * vec[a];
  len = sqrt(len);
  for(long long a = 0; a < m_nn->dim(); a++) vec[a] /= len;
}

real Model::doc_likelihood(TaggedDocument & doc, int skip)
{
  if(!m_hs){
    return 0;
  }
  TrainModelThread trainThread(0, this, NULL, true);
  trainThread.buildDocument(doc, skip);
  return trainThread.doc_likelihood();
}

real Model::context_likelihood(TaggedDocument & doc, int sentence_position)
{
  if(!m_hs){
    return 0;
  }
  if(m_word_vocab->searchVocab(doc.m_words[sentence_position]) == -1 ||
     m_word_vocab->searchVocab(doc.m_words[sentence_position]) == 0)
  {
    return 0;
  }
  TrainModelThread trainThread(0, this, NULL, true);
  trainThread.buildDocument(doc);

  long long sent_pos = sentence_position;
  for(int i = 0; i < sentence_position; i++)
  {
    long long word_idx = m_word_vocab->searchVocab(doc.m_words[i]);
    if (word_idx == -1) sent_pos--;
  }
  return trainThread.context_likelihood(sent_pos);
}

void Model::save(FILE * fout) const
{
  m_word_vocab->save(fout);
  m_doc_vocab->save(fout);
  m_nn->save(fout);

  int cbow = m_cbow, hs = m_hs;
  
  fwrite(&cbow, sizeof(int), 1, fout);
  fwrite(&hs, sizeof(int), 1, fout);
  fwrite(&m_negative, sizeof(int), 1, fout);
  fwrite(&m_window, sizeof(int), 1, fout);
  fwrite(&m_start_alpha, sizeof(real), 1, fout);
  fwrite(&m_sample, sizeof(real), 1, fout);
  fwrite(&m_iter, sizeof(int), 1, fout);
  m_wmd->save(fout);
}

void Model::load(FILE * fin)
{
  m_word_vocab = std::make_unique<Vocabulary>();
  m_word_vocab->load(fin);
  
  m_doc_vocab = std::make_unique<Vocabulary>();
  m_doc_vocab->load(fin);

  m_nn = std::make_unique<NN>();
  m_nn->load(fin);

  int cbow, hs;
  fread(&cbow, sizeof(int), 1, fin);
  fread(&hs, sizeof(int), 1, fin);
  fread(&m_negative, sizeof(int), 1, fin);
  fread(&m_window, sizeof(int), 1, fin);
  fread(&m_start_alpha, sizeof(real), 1, fin);
  fread(&m_sample, sizeof(real), 1, fin);
  fread(&m_iter, sizeof(int), 1, fin);

  m_cbow = cbow;
  m_hs = hs;
  
  initNegTable();
  m_nn->norm();

  m_wmd = std::make_unique<WMD>(this);
  m_wmd->load(fin);
}

size_t Model::dim() const { return m_nn->dim(); }
