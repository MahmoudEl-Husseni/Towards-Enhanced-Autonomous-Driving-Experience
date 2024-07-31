/*
 * pid.h
 *
 *  Created on: Mar 1, 2024
 *      Author: engmu
 */

#ifndef DC_MOTOR_PID_PID_H_
#define DC_MOTOR_PID_PID_H_

/* ----- Includes :  Section  ------ */

#include "../../Util/util.h"
#include "../dc_motor.h"

/* ----- Macros  :  Section  ------ */

#define MAX_SPEED 10000

/* ----- User Defined Data Types  :  Section  ------ */
typedef enum {
	PID_OK ,
	PID_NOT_OK

}PID_status_t;

/* ----- User interface functions   :  Section  ------ */
std_return_type PID (int16_t target_position,
					 int16_t actual_position,
					 int32_t *pid_res) ;


#endif /* DC_MOTOR_PID_PID_H_ */
