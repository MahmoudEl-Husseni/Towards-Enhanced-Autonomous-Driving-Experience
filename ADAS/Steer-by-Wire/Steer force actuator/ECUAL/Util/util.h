/*
 * util.h
 *
 *  Created on: Feb 19, 2024
 *      Author: engmu
 */

#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_


/* ------Section :Includes ------ */

#include "stm32f1xx_hal.h"

#include <stdlib.h>
#include <math.h>

/* ------Section :Macros  ------ */

#define HIGH  1
#define LOW   0
#define ENDS_ERROR 15
#define TRUE  1
#define FALSE 0

/* -----	Bit Math MAcros ----	*/
#define ST_BIT(reg,bit)  reg|=(1<<bit)
#define CLR_BIT(reg,bit)     reg&=~(1<<bit)
#define GIT_BIT(reg, bit)   ((reg >> bit) & (1))
#define TOGGLE_BIT(reg,bit)    reg^=(1<<bit)


/* ------Section : User Defined Data Types ------*/

typedef uint8_t std_return_type ;  // <--  Standard function return Types for Error Handler //


typedef struct {
	float input_min_range  ;
	float input_max_range  ;
	float output_min_range  ;
	float output_max_range  ;
} map_range_t ;

typedef enum {
	MAP_OK ,
	MAP_NOT_OK,
	__ROUND_RANGE_OK ,
	__ROUND_RANGE_NOT_OK ,
	__ROUND_OK,
	__ROUND_NOT_OK

}errors_t;


/*---------- Section : Function declaration  ----------- */
std_return_type   __round__ (float input , int32_t * output   ) ;

std_return_type in_between (int16_t _min , int16_t _max ,
							int16_t input , uint8_t *output  ) ;

std_return_type   __round_range  (int32_t input , int32_t * output ,
										uint16_t resoultion  ) ;

std_return_type map (float input , float *output ,
					 map_range_t *ranges) ;
#endif /* UTIL_UTIL_H_ */
