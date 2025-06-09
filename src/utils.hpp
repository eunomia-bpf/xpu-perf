#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <memory>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

#include <linux/types.h>
#include "../blazesym/target/release/blazesym.h"

#ifdef __cplusplus
}
#endif

// Common constants
#define TASK_COMM_LEN 16
#define MAX_PID_NR 30
#define MAX_TID_NR 30

// Forward declaration
struct blazesym;

// Common RAII deleter for blazesym
struct BlazesymDeleter {
    void operator()(struct blazesym* sym) const;
};

// Common utility functions
void show_stack_trace(struct blazesym *symbolizer, __u64 *stack, int stack_sz, pid_t pid);
void show_stack_trace_folded(struct blazesym *symbolizer, __u64 *stack, int stack_sz, 
                            pid_t pid, char separator, bool reverse);

// String utilities
int split_convert(char *s, const char* delim, void *elems, size_t elems_size,
                  size_t elem_size, int (*convert)(const char *, void *));
int str_to_int(const char *src, void *dest);
char *safe_strdup(const char *s);

#endif /* __UTILS_HPP */ 