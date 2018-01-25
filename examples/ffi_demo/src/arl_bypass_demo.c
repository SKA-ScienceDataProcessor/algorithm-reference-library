
/*
 * Compile with:
 *     gcc -Wall -Wextra arl_bypass_demo.c -o arl_bypass_demo
 *
 * Test with:
 *     ARL_BYPASS=0 ./arl_bypass_demo
 *     ARL_BYPASS=1 ./arl_bypass_demo
 */


/** In arl_bypass.h: *********************************************************/

/**
 * Returns a flag to specify whether calls to ARL should actually happen.
 *
 * Set the environment variable ARL_BYPASS=TRUE or ARL_BYPASS=1
 * to skip calls to actual ARL functions using the wrapper.
 */
int arl_bypass(void);


/** In arl_bypass.c: *********************************************************/

#include <stdlib.h>
#include <string.h>
/*#include "arl_bypass.h"*/

/* Thread-local, file-local (global) variables.
 * These are not accessed directly in user code. */
#ifdef __GNUC__
#define ARL_TLS __thread
#else
#define ARL_TLS
#endif
static ARL_TLS int arl_bypass_ = 0;
static ARL_TLS int arl_bypass_check_ = 0;


int arl_bypass(void)
{
    /* Return immediately if the environment variable
     * has already been checked. */
    if (!arl_bypass_check_)
    {
        char* flag = getenv("ARL_BYPASS");
        arl_bypass_check_ = 1;
        if (flag && (
                !strcmp(flag, "TRUE") ||
                !strcmp(flag, "true") ||
                !strcmp(flag, "1") ))
            arl_bypass_ = 1;
    }
    return arl_bypass_;
}


/** In arl_wrapper_function.c: ***********************************************/

/* For sleep() function: */
#include <unistd.h>
#include <stdio.h>
/*#include "arl_bypass.h"*/
/*#include "arl_wrapper_function.h"*/

void arl_wrapper_function(void)
{
    /* Return immediately if ARL is bypassed. */
    if (arl_bypass()) return;

    /* Normal wrapper code. */
    printf("Calling ARL Python function.\n");
    sleep(1);
}


/** In main.c: ***************************************************************/

/*#include "arl_wrapper_function.h"*/
int main()
{
    printf("Calling arl_wrapper_function()...\n");
    arl_wrapper_function();
    return 0;
}
