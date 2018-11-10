#include "nfa.h"
#include "nfa_device.h"

 __device__ List l1, l2;
 __device__ static int listid;

/* Compute initial state list */
__device__ List* startlist(State *start, List *l) {
  l->n = 0;
  listid++;
  addstate(l, start);
  return l;
}

/* Check whether state list contains a match. */
__device__ int ismatch(List *l, State* matchstate) {
  int i;

  for(i=0; i<l->n; i++)
	if(l->s[i] == matchstate)
	  return 1;
  return 0;
}

/* Add s to l, following unlabeled arrows. */
__device__ void addstate(List *l, State *s) {
  if(s == NULL || s->lastlist == listid)
	return;
  s->lastlist = listid;
  if(s->c == Split){
	/* follow unlabeled arrows */
	addstate(l, s->out);
	addstate(l, s->out1);
	return;
  }
  l->s[l->n++] = s;
}

/*
 * Step the NFA from the states in clist
 * past the character c,
 * to create next NFA state set nlist.
 */
__device__ void step(List *clist, int c, List *nlist) {
  int i;
  State *s;

  listid++;
  nlist->n = 0;
  for(i=0; i<clist->n; i++){
	s = clist->s[i];
	if(s->c == c)
	  addstate(nlist, s->out);
  }
}

/* Run NFA to determine whether it matches s. */
__global__ int match(State *start, char *s, State* matchstate) {
  int i, c;
  List *clist, *nlist, *t;

  clist = startlist(start, &l1);
  nlist = &l2;
  for(; *s; s++){
	c = *s & 0xFF;
	step(clist, c, nlist);
	t = clist; clist = nlist; nlist = t;	/* swap clist, nlist */
  }
  return ismatch(clist, matchstate);
}

