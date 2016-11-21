/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack;
data_t data;
pthread_barrier_t barr;


#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
        // just call pop
        stack_pop(stack);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);
    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;
  stack_element_t new_ele;
  new_ele.value = (unsigned int)pthread_self();
  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
      // See how fast your implementation can push MAX_PUSH_POP elements in parallel
      // just call push with the a new element
      stack_push(stack,&new_ele);
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);
  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));
  stack->head = NULL; // empty so should not point anywhere
  // Reset explicitely all members to a well-known initial value
  // initialize the stack with predefined data
  int i;
  stack_element_t* tmp;
  for (i = 0; i < MAX_PUSH_POP; ++i)
    {
      tmp = malloc(sizeof(stack_element_t));
      tmp->value = i;
      stack_push(stack,tmp);
    }
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  // just iterate through the stack and free all.
  stack_element_t* tmp;
  while (stack->head != NULL)
    {
      tmp = stack_pop(stack);
      free(tmp);
    }
  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it
  int i;
  // get the thread id
  unsigned int start, end, iterations = 20;
  unsigned int tid = (unsigned int)pthread_self();
  // add some constant dependent on tid
  start = (unsigned int)tid * 100;
  end = start + iterations;
  for (i = 0; i < iterations; ++i)
    {
      stack_element_t* new_ele = malloc(sizeof(stack_element_t));
      new_ele->value = start+i;
      stack_push(stack, new_ele);
    }

  // check if the stack is in a consistent state
  stack_check(stack);
  //printf("My thread id: %d \n", tid);
  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // iterate through the stack and see if there are 20 values with the threads id

  int result;
  unsigned int counter = 0;
  stack_element_t* current_ele = stack->head;
  for (i = 0; i < iterations * NB_THREADS; ++i)
    {
      // if we have not pushed all elements to the stack (race condition)
      if (current_ele == NULL)
        {
          result = 0;
        }
      // if we find a value added by this thread increment counter
      else if (current_ele->value >= start && current_ele->value <= end)
        {
          counter++;
          if (counter == iterations)
            {
              result = 1;
            }
        }
      current_ele = current_ele->next;
    }
  // assert(stack->head->value == 0);

  return result;
}

int
test_pop_safe()
{
  int i;
  // get the thread id
  unsigned int start, end, iterations = 20;
  unsigned int tid = (unsigned int)pthread_self();
  // add some constant dependent on tid
  start = (unsigned int)tid * 100;
  end = start + iterations;
  stack_element_t* tmp;
  for (i = 0; i < iterations; ++i)
    {
      tmp = stack_pop(stack);
      free(tmp);
    }
  // check if the stack is in a consistent state
  stack_check(stack);
  //printf("My thread id: %d \n", tid);
  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // iterate through the stack and see if there are 0 values with the threads id

  int result = 1;
  stack_element_t* current_ele = stack->head;
  while (current_ele != NULL)
    {
      // if we find a value added by this thread increment counter
      if (current_ele->value >= start && current_ele->value <= end)
        {
          result = 0;
          break;
        }
      current_ele = current_ele->next;
    }

  return result;
}

// 2 Threads should be enough to raise and detect the A0BA problem
#define ABA_NB_THREADS 2

struct thread_test_aba_args_t {
  unsigned int id;
};
typedef struct thread_test_aba_args thread_test_aba_args_t;

void
thread_test_aba(void* arg)
{
  struct thread_test_aba_args_t* args = (struct thread_test_aba_args_t*) arg;
  // aquire a lock each (could have used semaphores, would have been cleaner)
  if (args->id == 0)
    {
      pthread_mutex_lock(&aba_mutex1);
    }
  else
    {
      pthread_mutex_lock(&aba_mutex2);
    }
  // wait for all threads
  int rc = pthread_barrier_wait(&barr);
	if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf("Could not wait on barrier\n");
      exit(-1);
    }
  //   the stack has top -> 1 -> 2 -> 3
  //    Thread 1 starts pop(), gets preemted before cas(), has ret = 1 and next = 2
  //    Thread 2 runs pop() without getting preemted, the stack is now top -> 2 -> 3
  //    Thread 2 runs pop() again without getting preemted, the stack is now top -> 3
  //    Thread 2 pushes 1 back to the stack without getting preemted, the stack is now top -> 1 -> 3
  //    Thread 1 is allowed to run now and compares  1 == 1, which will pass as correct.
  if (args->id == 0)
    {
      // thread 1
      stack_element_t* tmp;
      tmp = aba_stack_pop(stack);
    }
  else
    {
      // thread 2
      pthread_mutex_lock(&aba_mutex1);
      stack_element_t *first,*second;
      first = stack_pop(stack);
      second = stack_pop(stack);
      stack_push(stack,first);
      pthread_mutex_unlock(&aba_mutex1);
      // allow thread 1 to run by releasing lock2
      pthread_mutex_unlock(&aba_mutex2);
    }
}

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success;
  // Write here a test for the ABA problem
  /*
    A test case with 2 threads sharing a resource which is modified back and forth.
    Will trigger the ABA problem
   */
  // Fill the stack with 1,2,3 sequentially
  int i;
  for (i = 1; i < 4; ++i)
    {
      stack_element_t* new_ele = malloc(sizeof(stack_element_t));
      new_ele->value = i;
      stack_push(stack, new_ele);
    }
  // create the threads
  pthread_t thread[ABA_NB_THREADS];
  pthread_attr_t attr;
  struct thread_test_aba_args_t args[ABA_NB_THREADS];
  pthread_mutexattr_t mutex_attr;

  test_setup();
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  // initialize the aba mutexes and barrier defined in stack.h
  pthread_mutex_init(&aba_mutex1, &mutex_attr);
  pthread_mutex_init(&aba_mutex2, &mutex_attr);
  pthread_barrier_init(&barr, NULL, 2);
  for (i = 0; i < ABA_NB_THREADS; i++)
    {
      args[i].id = i;
      pthread_create(&thread[i], &attr, &thread_test_aba, (void*) &args[i]);
    }
  // join all threads
  for (i = 0; i < ABA_NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  if (stack->head != NULL && stack->head->value == 3)
    {
      // implementation is affected by aba
      success = 0;
    }

  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);
  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];

  test_setup();
  pthread_attr_init(&attr);

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}
