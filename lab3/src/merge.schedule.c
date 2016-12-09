#include <stdlib.h> 

#include <drake/schedule.h> 

#include <drake/platform.h> 

#define TASK_NAME leaf_1
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_1 1

#define TASK_NAME leaf_2
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_2 1

#define TASK_NAME root
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_root 1

int drake_task_number()
{
	return 3;
}

char* drake_task_name(size_t index)
{
	switch(index - 1)
	{
		case 0:
			return "leaf_1";
		break;
		case 1:
			return "leaf_2";
		break;
		case 2:
			return "root";
		break;
		default:
			return "invalid task id";
		break;
	}
}

void drake_schedule_init(drake_schedule_t* schedule)
{
	schedule->core_number = 1;
	schedule->task_number = 3;
	schedule->stage_time = 0;

	schedule->tasks_in_core = malloc(sizeof(size_t) * schedule->core_number);

	schedule->task_name = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->task_name[0] = "leaf_1";
	schedule->task_name[1] = "leaf_2";
	schedule->task_name[2] = "root";


	schedule->task_workload = malloc(sizeof(double) * schedule->task_number);
	schedule->task_workload[0] = 1;
	schedule->task_workload[1] = 1;
	schedule->task_workload[2] = 1;

	schedule->tasks_in_core[0] = 3;

	schedule->consumers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->consumers_in_core[0] = 0;

	schedule->producers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->producers_in_core[0] = 0;

	schedule->consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->consumers_in_task[0] = 1;
	schedule->consumers_in_task[1] = 1;
	schedule->consumers_in_task[2] = 0;

	schedule->producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->producers_in_task[0] = 0;
	schedule->producers_in_task[1] = 0;
	schedule->producers_in_task[2] = 2;

	schedule->remote_consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_consumers_in_task[0] = 0;
	schedule->remote_consumers_in_task[1] = 0;
	schedule->remote_consumers_in_task[2] = 0;

	schedule->remote_producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_producers_in_task[0] = 0;
	schedule->remote_producers_in_task[1] = 0;
	schedule->remote_producers_in_task[2] = 0;

	schedule->producers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->producers_rate = malloc(sizeof(size_t**) * schedule->task_number);
	schedule->producers_name = malloc(sizeof(char*) * schedule->task_number);
	schedule->producers_id[0] = NULL;
	schedule->producers_rate[0] = NULL;
	schedule->producers_name[0] = NULL;
	schedule->producers_id[1] = NULL;
	schedule->producers_rate[1] = NULL;
	schedule->producers_name[1] = NULL;
	schedule->producers_id[2] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[2][0] = 1;
	schedule->producers_id[2][1] = 2;
	schedule->producers_rate[2] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[2][0] = 1;
	schedule->producers_rate[2][1] = 1;
	schedule->producers_name[2] = malloc(sizeof(char*) * 2);
	schedule->producers_name[2][0] = "output";
	schedule->producers_name[2][1] = "output";

	schedule->consumers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->consumers_rate = malloc(sizeof(size_t**) * schedule->task_number);
	schedule->consumers_name = malloc(sizeof(char*) * schedule->task_number);
	schedule->consumers_id[0] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[0][0] = 3;
	schedule->consumers_rate[0] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[0][0] = 1;
	schedule->consumers_name[0] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[0][0] = "left";
	schedule->consumers_id[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[1][0] = 3;
	schedule->consumers_rate[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[1][0] = 1;
	schedule->consumers_name[1] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[1][0] = "right";
	schedule->consumers_id[2] = NULL;
	schedule->consumers_rate[2] = NULL;
	schedule->consumers_name[2] = NULL;

	schedule->schedule = malloc(sizeof(drake_schedule_task_t*) * schedule->core_number);
	schedule->schedule[0] = malloc(sizeof(drake_schedule_task_t) * 3);
	schedule->schedule[0][0].id = 2;
	schedule->schedule[0][0].start_time = 0;
	schedule->schedule[0][0].frequency = 16;
	schedule->schedule[0][1].id = 1;
	schedule->schedule[0][1].start_time = 2.86e-07;
	schedule->schedule[0][1].frequency = 16;
	schedule->schedule[0][2].id = 3;
	schedule->schedule[0][2].start_time = 5.71e-07;
	schedule->schedule[0][2].frequency = 16;
}

void drake_schedule_destroy(drake_schedule_t* schedule)
{
	free(schedule->schedule[0]);

	free(schedule->schedule);
	free(schedule->consumers_id[0]);
	if(schedule->consumers_rate[0] != NULL)
	{
		free(schedule->consumers_rate[0]);
	}
	free(schedule->consumers_name[0]);
	free(schedule->consumers_id[1]);
	if(schedule->consumers_rate[1] != NULL)
	{
		free(schedule->consumers_rate[1]);
	}
	free(schedule->consumers_name[1]);
	free(schedule->consumers_id[2]);
	if(schedule->consumers_rate[2] != NULL)
	{
		free(schedule->consumers_rate[2]);
	}
	free(schedule->consumers_name[2]);
	free(schedule->consumers_id);
	free(schedule->consumers_rate);
	free(schedule->consumers_name);

	free(schedule->producers_id[0]);
	if(schedule->producers_rate[0] != NULL)
	{
		free(schedule->producers_rate[0]);
	}
	free(schedule->producers_name[0]);
	free(schedule->producers_id[1]);
	if(schedule->producers_rate[1] != NULL)
	{
		free(schedule->producers_rate[1]);
	}
	free(schedule->producers_name[1]);
	free(schedule->producers_id[2]);
	if(schedule->producers_rate[2] != NULL)
	{
		free(schedule->producers_rate[2]);
	}
	free(schedule->producers_name[2]);
	free(schedule->producers_id);
	free(schedule->producers_rate);
	free(schedule->producers_name);
	free(schedule->task_workload);
	free(schedule->remote_producers_in_task);
	free(schedule->remote_consumers_in_task);
	free(schedule->producers_in_task);
	free(schedule->consumers_in_task);
	free(schedule->producers_in_core);
	free(schedule->consumers_in_core);
	free(schedule->tasks_in_core);
	free(schedule->task_name);
}

size_t
drake_task_width(task_tp task){
	size_t task_width[3] = {1, 1, 1, };
	return task_width[task->id - 1];
}

size_t
drake_core_id(task_tp task){
	size_t local_core_id[1][3] = {
			{0, 0, 0, },
		};
	return local_core_id[drake_platform_core_id()][task->id - 1];
}

void*
drake_function(size_t id, task_status_t status)
{
	switch(id)
	{
		default:
			// TODO: Raise an alert
		break;
		case 1:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_1);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_1);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_1);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_1);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_1);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 2:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_2);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_2);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_2);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_2);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_2);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 3:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, root);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, root);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, root);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, root);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, root);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;

	}

	return 0;
}
