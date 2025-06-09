#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <memory>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <linux/types.h>
#include "blazesym.h"

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
    void operator()(struct blazesym* sym) const {
        if (sym) {
            blazesym_free(sym);
        }
    }
};

// Common utility functions
inline void show_stack_trace(struct blazesym *symbolizer, __u64 *stack, int stack_sz, pid_t pid);
inline void show_stack_trace_folded(struct blazesym *symbolizer, __u64 *stack, int stack_sz, 
                            pid_t pid, char separator, bool reverse);

// String utilities
inline int split_convert(char *s, const char* delim, void *elems, size_t elems_size,
                  size_t elem_size, int (*convert)(const char *, void *));
inline int str_to_int(const char *src, void *dest);
inline char *safe_strdup(const char *s);

// Implementation section

/**
 * safe_strdup - Safe string duplication
 * @s: Source string
 *
 * Return: Newly allocated string copy or NULL on failure
 */
inline char *safe_strdup(const char *s)
{
    if (!s) {
        return NULL;
    }
    
    size_t len = strlen(s) + 1;
    char *copy = (char *)malloc(len);
    if (!copy) {
        return NULL;
    }
    
    return strcpy(copy, s);
}

/**
 * split_convert - Split a string by a delimiter and convert each token
 * @s: String to split
 * @delim: Delimiter string
 * @elems: Array to store the converted elements
 * @elems_size: Size of the elems array in bytes
 * @elem_size: Size of each element in bytes
 * @convert: Function to convert each token to the desired type
 *
 * Return: 0 on success, negative error code on failure
 */
inline int split_convert(char *s, const char* delim, void *elems, size_t elems_size,
                  size_t elem_size, int (*convert)(const char *, void *))
{
    char *token;
    int ret;
    char *pos = (char *)elems;

    if (!s || !delim || !elems)
        return -1;

    token = strtok(s, delim);
    while (token) {
        if (pos + elem_size > (char*)elems + elems_size)
            return -ENOBUFS;

        ret = convert(token, pos);
        if (ret)
            return ret;

        pos += elem_size;
        token = strtok(NULL, delim);
    }

    return 0;
}

/**
 * str_to_int - Convert a string to an integer
 * @src: Source string
 * @dest: Pointer to store the converted integer
 *
 * Return: 0 on success, negative error code on failure
 */
inline int str_to_int(const char *src, void *dest)
{
    *(int*)dest = strtol(src, NULL, 10);
    return 0;
}

#endif /* __UTILS_HPP */ 