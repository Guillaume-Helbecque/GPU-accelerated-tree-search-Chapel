#ifndef PFSP_NODE_H
#define PFSP_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_SIZE 512

#define MAX_JOBS 20
#define MAX_MACHINES 20

typedef struct
{
  int depth;
  int limit1;
  int prmu[MAX_JOBS];
} Node;

void initRoot (Node *root, const int jobs);

#ifdef __cplusplus
}
#endif

#endif // PFSP_NODE_H
