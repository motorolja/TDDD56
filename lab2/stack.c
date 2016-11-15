/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

void
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
}

void /* Return the type you prefer */
stack_push(stack_t *s, stack_element_t *e)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&s->glock);
  e->next = s->head;
  s->head = e;
  pthread_mutex_unlock(&s->glock);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  size_t result;
  stack_element_t* old;
  do
    {
      // get current head
      old = s->head;
      // set new element to point towards current head
      e->next = s->head;
      // do Compare-And-Swap hardware instruction and save the result, ensure that they are the right types/sizes
      result = cas((size_t *)& s->head, (size_t) old, (size_t) e);
    } while (result != (size_t) old);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
  // pretty much the same as above but with atomic around what is suppose to happen in CAS
  // ATOMIC();

  // END_ATOMIC();
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);
}

stack_element_t* /* Return the type you prefer */
stack_pop(stack_t *s)
{
  stack_element_t* popped;
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&s->glock);
  // save the head before poping
  popped = s->head;
  s->head = s->head->next;
  pthread_mutex_unlock(&s->glock);
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  stack_element_t* old;
  size_t result;
  do
    {
      // get current head
      old = s->head;
      // set popped element to point towards current head
      popped = s->head;
      // do Compare-And-Swap hardware instruction and save the result, ensure that they are the right types/sizes.
      result = cas((size_t *)& s->head, (size_t) old, (size_t) popped->next);
    } while (result != (size_t) old);
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
  // pretty much the same as above but with atomic around what is suppose to happen in CAS
  // ATOMIC();

  // END_ATOMIC();
#endif
  return popped;
}

