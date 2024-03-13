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

/* ------Section :Macros  ------ */

#define HIGH  1
#define LOW   0

/* -----	Bit Math MAcros ----	*/
#define ST_BIT(reg,bit)  reg|=(1<<bit)
#define CLR_BIT(reg,bit)     reg&=~(1<<bit)
#define GIT_BIT(reg, bit)   ((reg >> bit) & (1))
#define TOGGLE_BIT(reg,bit)    reg^=(1<<bit)


/* ------Section : User Defined Data Types ------*/

typedef uint8_t std_return_type ;  // <--  Standard function return Types for Error Handler //



#endif /* UTIL_UTIL_H_ */
