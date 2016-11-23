#include <cstdio>
#include <algorithm>

#include <string.h>
#include <cmath>
#include "sort.h"

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compareile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
int *begin;
#define debug(var) printf("[%s:%s:%d] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#else
#define show(first, last)
#define show_ptr(first, last)
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray
{
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	public:
	DynArray(T* buffer, size_t size)
	{
		this->buffer = buffer;
		this->size = size;
	}

	iterator begin()
	{
		return buffer;
	}

	iterator end()
	{
		return buffer + size;
	}

	protected:
		T* buffer;
		size_t size;
};

static
void
cxx_sort(int *array, size_t size)
{
	DynArray<int> cppArray(array, size);
	std::sort(cppArray.begin(), cppArray.end());
}

// could do a random picking as well but oh well, this should be good enough
size_t quicksort_pick_pivot(int *array, size_t size, size_t offset)
{
  size_t result = 0;
  // if we only have 2 elements just pick the first.
  if ( size > 2)
    {
      // compare first, middle and last elements and swap them to be internaly sorted
      size_t first = offset, middle = (size/2) + offset, last = size - 1 + offset;
      // The result is the median value of the three but it will always be in the middle
      result = middle;
      if (array[first] > array[last])
        {
          std::swap(array[first], array[last]);
        }
      if (array[middle] > array[last])
        {
          std::swap(array[middle], array[last]);
        }
      if (array[first] > array[middle])
        {
          std::swap(array[first], array[middle]);
        }
    }

  return result;
}

// A very simple quicksort implementation
// * Recursion until array size is 1
static void sequential_quicksort(int *array, size_t size, size_t offset = 0)
{
	int pivot, pivot_count, i;
	// int *left, *right;
	size_t left_size = 0, right_size = 0;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(size > 1)
	{
    pivot = quicksort_pick_pivot(array, size, offset);

    // left = (int*)malloc(size * sizeof(int));
		// right = (int*)malloc(size * sizeof(int));

		// Split
		for(i = offset; i < size + offset; i++)
		{
			if(array[i] < array[pivot])
			{
				left_size++;
			}
			else if(array[i] > array[pivot])
			{
        std::swap(array[i],array[pivot]);
        pivot = i;
				right_size++;
			}
			else
			{
				pivot_count++;
			}
		}

		// Recurse
		sequential_quicksort(array, left_size, offset);
		sequential_quicksort(array, right_size, right_size + pivot_count + offset);

		//// Merge
		//memcpy(array, left, left_size * sizeof(int));
		//for(i = left_size; i < left_size + pivot_count; i++)
		//{
		//	array[i] = pivot;
		//}
		//memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

		//// Free
		//free(left);
		//free(right);
	}
	else
	{
		// Do nothing
	}
}

struct quicksort_args_t
{
  int* array;
  size_t size;
  size_t depth = 1;
  size_t offset = 0;
  int id;
};

static void parallel_quicksort(void* arg)
{
  quicksort_args_t *args = (quicksort_args_t*)arg;
  int pivot, pivot_count, i;
	// int *left, *right;
	size_t left_size = 0, right_size = 0, depth = args->depth;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(args->size > 1)
    {
      pivot = quicksort_pick_pivot(args->array, args->size, args->offset);

      // Split
      for(i = args->offset; i < args->size + args->offset; i++)
        {
          if(args->array[i] < args->array[pivot])
            {
              left_size++;
            }
          else if(args->array[i] > args->array[pivot])
            {
              std::swap(args->array[i],args->array[pivot]);
              pivot = i;
              right_size++;
            }
          else
            {
              pivot_count++;
            }
        }

      // Recurse

      // This is to ensure unique thread ids, does not need to be created in the order
      // which the ids exists! e.g thread id 8 could be created before 5 and so on.
      size_t next_tid = pow(2,(depth-1)) + args->id;
      depth++;
      // means we will get new threads on depth 1 and 2, in other words:
      // 2 threads depth 1
      // 4 threads depth 2
      // 8 threads at depth 3 --- this is more than we can run at the same time!
      if (NB_THREADS > 1 && next_tid <= NB_THREADS && depth <= 3)
        {
          quicksort_args_t *left_args = new quicksort_args_t();
          left_args->array = args->array;
          left_args->size = left_size;
          left_args->depth = depth;
          left_args->offset = args->offset;
          left_args->id = args->id;

          quicksort_args_t *right_args = new quicksort_args_t();
          right_args->array = args->array;
          right_args->size = right_size;
          right_args->depth = depth;
          right_args->offset = right_size + pivot_count + args->offset;
          right_args->id = next_tid;

          pthread_t thread_right;
          pthread_attr_t attr;

          printf("Started new thread %d, at depth %d\n", next_tid, depth);
          pthread_create(&thread_right, &attr, parallel_quicksort, (void*)right_args);
          parallel_quicksort((void*)left_args);

          pthread_join(thread_right, NULL);

          delete(left_args);
          delete(right_args);
         }
      else
        {
          sequential_quicksort(args->array, left_size);
          sequential_quicksort(args->array, right_size, right_size + pivot_count + args->offset);
        }
   }
}

// This is used as sequential sort in the pipelined sort implementation with drake (see merge.c)
// to sort initial input data chunks before streaming merge operations.
void
sort(int* array, size_t size)
{
	// Do some sorting magic here. Just remember: if NB_THREADS == 0, then everything must be sequential
	// When this function returns, all data in array must be sorted from index 0 to size and not element
	// should be lost or duplicated.

	// Use preprocessor directives to influence the behavior of your implementation. For example NB_THREADS denotes
	// the number of threads to use and is defined at compareile time. NB_THREADS == 0 denotes a sequential version.
	// NB_THREADS == 1 is a parallel version using only one thread that can be useful to monitor the overhead
	// brought by addictional parallelization code.


	// Alternatively, use C++ sequential sort, just to see how fast it is
	//cxx_sort(array, size);

	// Note: you are NOT allowed to demonstrate code that uses C or C++ standard sequential or parallel sort or merge
	// routines (qsort, std::sort, std::merge, etc). It's more interesting to learn by writing it yourself.



	// Reproduce this structure here and there in your code to compile sequential or parallel versions of your code.
#if NB_THREADS == 0
	// Some sequential-specific sorting code
  sequential_quicksort(array, size);
#else
	// Some parallel sorting-related code
  quicksort_args_t *args = new quicksort_args_t();
  args->size = size;
  args->array = array;
  args->depth = 0;
  // this is the main thread, e.g. id = 1;
  args->id = 1;

  parallel_quicksort(args);
  delete(args);
#endif // #if NB_THREADS
}

