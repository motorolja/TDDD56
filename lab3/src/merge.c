#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <drake.h>
#include <drake/link.h>
#include <drake/eval.h>
#include <pelib/integer.h>

#include "sort.h"
#include "utils.h"

// Filename of file containing the data to sort 
static char *input_filename;

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
#define debug(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#else
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

int
drake_init(task_t *task, void* aux)
{
	link_t *link;
	array_t(int) *tmp;
	size_t input_buffer_size, input_size, i;

	// If a task has no consumer link then it is the root
	// task. Here, we build a new link where the root task
	// can write the final sorted array
	args_t *args = (args_t*)aux;
	if(pelib_map_size(string, link_tp)(task->succ) == 0 && args->argc > 0)
	{
		input_filename = ((args_t*)aux)->argv[0];
		array_t(int) *tmp = pelib_array_preloadfilenamebinary(int)(input_filename);
		input_size = pelib_array_length(int)(tmp);
		// Destroy the output links array
		pelib_free(map_t(string, link_tp))(task->succ);
		// Create a new output link array that can hold one link
		task->succ = pelib_alloc(map_t(string, link_tp))();
		pelib_init(map_t(string, link_tp))(task->succ);
		// Allocate a new link
		link = (link_t*)malloc(sizeof(link_t));
		// Set the size of the new link to the size of the complete input array
		size_t capacity = (int)ceil((double)input_size);
		// Allocate a big output fifo
		link->buffer = pelib_alloc_collection(cfifo_t(int))(capacity);
		// There is no actual consumer task in this link
		link->cons = NULL;
		// The producer task end of this link is the task being initialized
		link->prod = task;
		// Initialize the fifo: set its read and write pointer to the
		// beginning of the fifo buffer and mark the fifo as empty.
		pelib_init(cfifo_t(int))(link->buffer);
		// Add the fifo to the output link array
		pair_t(string, link_tp) link_prod_pair;
		string name = "output";
		pelib_alloc_buffer(string)(&link_prod_pair.key, (strlen(name) + 1) * sizeof(char));
		pelib_copy(string)(name, &link_prod_pair.key);
		pelib_copy(link_tp)(link, &link_prod_pair.value);
		pelib_map_insert(string, link_tp)(task->succ, link_prod_pair);
	}

	// Everything always goes fine, doesn't it?
	return 1;
}

int
drake_start(task_t *task)
{
	link_t *link;
	int j;

	// Do nothing
	// No need to run start again
	return 1;
}

int
drake_run(task_t *task)
{
	// Input and output links
	link_t *left_link, *right_link, *parent_link;

	// If a task has no input links or no output link, then it can do nothing
	if(pelib_map_size(string, link_tp)(task->pred) < 2 || pelib_map_size(string, link_tp)(task->succ) < 1)
	{
		// Terminate immediately
		return 1;
	}

	// Read input links from input link buffers
	left_link = pelib_map_read(string, link_tp)(pelib_map_find(string, link_tp)(task->pred, "left")).value;
	right_link = pelib_map_read(string, link_tp)(pelib_map_find(string, link_tp)(task->pred, "right")).value;
	parent_link = pelib_map_read(string, link_tp)(pelib_map_find(string, link_tp)(task->succ, "output")).value;

	// Fetch buffer addresses
	size_t left_size = 0, right_size = 0, parent_size = 0, left_consumed = 0, right_consumed = 0, parent_pushed = 0;
	int *left, *right, *parent;

	parent = pelib_cfifo_writeaddr(int)(parent_link->buffer, &parent_size, NULL);
	left = pelib_cfifo_peekaddr(int)(left_link->buffer, 0, &left_size, NULL);
	right = pelib_cfifo_peekaddr(int)(right_link->buffer, 0, &right_size, NULL);

	// left is a pointer to data received from left child (contains left_size elements)
	// right is a pointer to data received from left child (contains right_size elements)
	// parent is a pointer to communication buffer toward parent (can hold up to parent_size element /!\ left_size + right_size > parent_size)


	// Merge as much as you can here


	// Don't forget, the task may receive more data from its left or right child, unless the left or right child terminated.
	// You will need to know the state of a task with
	//
	// drake_task_killed(task)
	//
	// where task is a task descriptor. Parameter task of this function is the descriptor of the task running. Left child task descriptor
	// is in left_link->pred and I let you guess where is the descriptor for the right child. You shouldn't need the descriptor of the
	// parent task.
	// This returns 0 is more data can be accessible in later iterations, 1 if no more input can be expected from the task or if the task
	// will not receive any more input from any of its input channels.

	
	// Write the number of element you consumed from left child and right child into left_consumed and right_consumed, respectively
	// and the total number of elements you pushed toward parent in parent_pushed


	// Now discarding input consumed and pushed output produced through channels and using the number of elements consumed and produced
	// that you set above.
	pelib_cfifo_discard(int)(left_link->buffer, left_consumed);
	pelib_cfifo_discard(int)(right_link->buffer, right_consumed);
	pelib_cfifo_fill(int)(parent_link->buffer, parent_pushed);

	// Finally, tell drake if this task should run more iterations (if it may have more input to process or data to push), or if it can
	// terminate now. If the task should continue, then return 0. If the task should stop now, then return 1.
	// Help yourself with:
	//
	// check drake_task_is_depleted(task_tp t)
	//
	// That returns 1 if all predecessors of task t are killed and all input buffers are empty, or if task t is killed and 0 otherwise.
	return 0;
}

int
drake_kill(task_t *task)
{
	// Everything went just fine. Task just got killed
	return 1;
}

int
drake_destroy(task_t *task)
{
	// If consumer task at the end of output link is NULL, then
	// task is the root task. We check if the output link of the
	// root task is sorted and is equivalent to the data in input
	// file.
	map_iterator_t(string, link_tp)* kk;
	kk = pelib_map_begin(string, link_tp)(task->succ);
	link_t *parent_link = pelib_map_read(string, link_tp)(kk).value;
	if(parent_link->cons == NULL)
	{
		check_sorted(input_filename, parent_link->buffer);
	}

	// Everything's fine, pipelined just got destroyed
	return 1;
}
