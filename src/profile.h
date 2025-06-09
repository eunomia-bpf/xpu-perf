// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#ifndef __PROFILE_H
#define __PROFILE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

/* Define BPF types for userspace if not already defined */
#ifndef __u32
typedef uint32_t __u32;
#endif

#define TASK_COMM_LEN		16
#define MAX_CPU_NR		128
#define MAX_ENTRIES		10240
#define MAX_PID_NR		30
#define MAX_TID_NR		30

struct profile_key_t {
	__u32 pid;
	int user_stack_id;
	int kern_stack_id;
	char name[TASK_COMM_LEN];
};

#endif /* __PROFILE_H */
