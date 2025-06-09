/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
#ifndef __OFFCPUTIME_H
#define __OFFCPUTIME_H

#define TASK_COMM_LEN		16
#define MAX_PID_NR		30
#define MAX_TID_NR		30

struct offcpu_key_t {
	unsigned int pid;
	unsigned int tgid;
	int user_stack_id;
	int kern_stack_id;
};

struct offcpu_val_t {
	unsigned long long delta;
	char comm[TASK_COMM_LEN];
};

#endif /* __OFFCPUTIME_H */
