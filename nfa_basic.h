#ifndef NFA_H
#define NFA_H 

extern int nstate;

/*
 * Represents an NFA state plus zero or one or two arrows exiting.
 * if c == Match, no arrows out; matching state.
 * If c == Split, unlabeled arrows to out and out1 (if != NULL).
 * If c < 256, labeled arrow with character c to out.
 */
enum
{
	Match = 256,
	Split = 257
};
typedef struct State State;
struct State
{
	int c;
	State *out;
	State *out1;
	int lastlist;
};

/*
 * A partially built NFA without the matching state filled in.
 * Frag.start points at the start state.
 * Frag.out is a list of places that need to be set to the
 * next state for this fragment.
 */
typedef struct Frag Frag;
typedef union Ptrlist Ptrlist;
struct Frag
{
  State *start;
  Ptrlist *out;
};

typedef struct List List;

struct List
{
  State **s;
  int n;
};

extern __device__ List l1, l2;

/*
 * Since the out pointers in the list are always
 * uninitialized, we use the pointers themselves
 * as storage for the Ptrlists.
 */
union Ptrlist
{
  Ptrlist *next;
  State *s;
};

/* Function declarations */
extern "C" char* re2post(char *re, char* buf);

extern "C" State* post2nfa(char *postfix, State* matchstate);

__device__ void addstate(List *l, State *s);

extern "C" __device__ int match(State *start, char *s, State* matchstate, int nstate);

#endif /* ifndef NFA_H */
