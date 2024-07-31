/*
 * pid.c
 *
 *  Created on: Mar 1, 2024
 *      Author: engmu
 */

#include "pid.h"

/*	=============- PID_Variables -================= */
static int32_t prev_error, error, diffrential, integral;

/*======= PID Weights ============*/
const uint32_t kp = 150, kd = 300, ki = 0;

/* ---- Function : Section 	------  */

/**
 * @brief  return value after compare input values with PID control.
 * @param  target_position position should reach.
 * @param  pid_res pointer to variable that will  .
 * @param  actual_position actual position reached
 * @retval PID_status_t
 */
int16_t counter;
int16_t delete_index;
int _counter_top;
static int16_t ki_queue[1000];

std_return_type PID(int16_t target_position, int16_t actual_position,
		int32_t *pid_res) {

	std_return_type ret = PID_NOT_OK;
	if (NULL == pid_res) {
		ret = PID_NOT_OK;
	} else {
		/*	=============- PID_Weights -================= */

		error = target_position - actual_position;

		diffrential = error - prev_error;
		if (prev_error != error) {
			ki_queue[(counter++)] = error ;
			integral += error;
		}
		integral += error;

//		ki_queue[counter++] = error ;
//
//		if (_counter_top >1000 )
//		{
//			integral -= ki_queue[((delete_index++)%1000)] ;
//
//			_counter_top= ((_counter_top + 1000) %100000) ;
//		}

		/*----- Calculate PID Result ------ */
		*pid_res = (error * kp) + (diffrential * kd) + (integral * ki);

		/*--------- limits the PID output in anti Clock Wise Direction  ------------*/
		if (*pid_res <= (-1 * MAX_SPEED)) {
			*pid_res = (-1 * MAX_SPEED);
		}
		/*--------- limits the PID output in  Clock Wise Direction  ------------*/

		else if (*pid_res > MAX_SPEED) {
			*pid_res = MAX_SPEED;
		}

		/*----- Save the Last Error ------ */
		prev_error = error;
		ret = PID_OK;

		counter %= 1000;
		_counter_top++;

	}
	return ret;
}
