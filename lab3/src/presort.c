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
#warning Activating debug output
#define debug(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#else
#warning Disabling debug output
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

static array_t(int) *local;

int
drake_init(task_t *task, void* aux)
{
	link_t *link;
	array_t(int) *tmp;
	size_t input_buffer_size, input_size, i;

	// The following makes tasks having no predecessor to load input data
	// and make a new input link that links to no task but that holds the
	// data to be sorted and merged

	// Fetch arguments and only load data if an input filename is given
	args_t *args = (args_t*)aux;
	if(args->argc > 0)
	{
		input_filename = ((args_t*)aux)->argv[0];

		// Read only the number of elements in input
		tmp = pelib_array_preloadfilenamebinary(int)(input_filename);
		if(tmp != NULL)
		{
			// Read the number of elements to be sorted
			input_size = pelib_array_length(int)(tmp);
			// No need of this array anymore
			pelib_free_struct(array_t(int))(tmp);

			size_t total_size;
			size_t chunk_size;
			size_t leaf_index;
			
			//Read the leaf task index from task name
			sscanf(task->name, "leaf_%zu", &leaf_index);

			// Calculate the number of elements each input links of this task will load, that is:
			// The total number of elements divided by the number of leaves. Here, we assume a
			// balanced binary tree.
			chunk_size = input_size / ((drake_task_number() + 1) / 2);

			// Load data
			local = pelib_array_loadfilenamewindowbinary(int)(input_filename, chunk_size * (leaf_index - 1), chunk_size);
		}
		else
		{
			fprintf(stderr, "[%s:%s:%d:P%zu:%s] Cannot open input file \"%s\". Check application arguments.\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, input_filename);
		}
	}
	else
	{
		fprintf(stderr, "[%s:%s:%d:P%zu:%s] Missing file to read input from. Check application arguments.\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name);
	}

	// Everything always goes fine, doesn't it?
	return 1;
}

int
drake_start(task_t *task)
{
	link_t *link;
	int j;

	// Locally sort local memory
	sort((int*)local->data, pelib_array_length(int)(local));

	// No need to run start again
	return 1;
}

int
drake_run(task_t *task)
{
	static size_t pushed = 0;
	size_t left = local->capacity - pushed;
	//drake_platform_time_get(run[task->id - 1]);

	link_t *parent_link;
	parent_link = pelib_map_read(string, link_tp)(pelib_map_find(string, link_tp)(task->succ, "output")).value;
	size_t parent_size = 0;
	int *parent;
	parent = pelib_cfifo_writeaddr(int)(parent_link->buffer, &parent_size, NULL);
	memcpy(parent, local->data + pushed, sizeof(int) * (parent_size < left ? parent_size : left));
	pelib_cfifo_fill(int)(parent_link->buffer, parent_size < left ? parent_size : left);

	pushed += parent_size < left ? parent_size : left;

	// Nothing else to do
	if(pushed < local->capacity)
	{
		return 0;
	}
	else
	{
		return 1;
	}
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
	// Everything's fine, pipelined just got destroyed
	return 1;
}

