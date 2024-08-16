#ifndef _QUEUE_H_
#define _QUEUE_H_

#include "parva_sched.h"

typedef struct node {
    void *data;
    struct node* next;
} Node;

typedef struct queue {
    Node* front;
    Node* rear;
} Queue;

Queue *initQueue();
void enqueue(Queue* q, void *data);
void *dequeue(Queue* q);
bool is_empty(Queue* q);

#endif