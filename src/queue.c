#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "queue.h"

Queue *initQueue() 
{
    Queue *q = malloc(sizeof(Queue));
    q->front = NULL;
    q->rear = NULL;
    return q;
}

bool is_empty(Queue* q)
{
    if (q->front == NULL && q->rear == NULL)
        return true;
    else
        return false;
}


void enqueue(Queue* q, void *data) 
{
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->next = NULL;
    
    if (q->rear == NULL) {
        q->front = q->rear = newNode;
        return;
    }
    
    q->rear->next = newNode;
    q->rear = newNode;
}

void *dequeue(Queue* q) 
{
    if (q->front == NULL) {
        return NULL;
    }
    
    Node* temp = q->front;
    void *data = temp->data;
    
    q->front = q->front->next;
    
    if (q->front == NULL) {
        q->rear = NULL;
    }
    
    free(temp);
    return data;
}