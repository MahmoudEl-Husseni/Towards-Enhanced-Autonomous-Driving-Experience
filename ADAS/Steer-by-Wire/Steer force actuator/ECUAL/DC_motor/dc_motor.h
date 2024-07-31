
/**
 * @brief .h file for the motor driver
 *
 *
 */

#ifndef INC_ECU_MOTOR_H_
#define INC_ECU_MOTOR_H_
/* ------ Section : includes -----   */
#include "stm32f1xx_hal.h"
#include "../Util/util.h"
/*	----- section : macros ------ */
#define LEFT_PORT 0
#define RIGHT_PORT 1

#define PWM 0
#define ENABLE 1

#define RESERVED 0xFFF

#define CPU_FREQ 72000000
#define PWM_FREQ 3000.0

/*	------ section : user defined data types------ */

typedef struct
{
	/*		User define only Ports and Pins 	*/
	GPIO_TypeDef *ports[2]; /*!< Select one Port for 2 Left and Right Pins */
	uint16_t left_pins[2];	/*!< select PWM and Enable Pins  */
	uint16_t right_pins[2]; /*!< Select PWM and Enabel Pins  */
	/* Motor speed*/
	uint8_t speed;

	/*				timers_init Function Put values in this 					*/

	/*************************************************************
	 * 					PWM Pins on left and right 				 *
	 *************************************************************/

	/*Left Pins */
	TIM_TypeDef *left_TIM_Instance;	  // timer 1 , timer 2 , timer 3 , timer4
	volatile uint32_t *left_TIM_CCRx; // channel -> ccrx

	/*right Pins */
	TIM_TypeDef *right_TIM_Instance;
	volatile uint32_t *right_TIM_CCRx;

} motor_t;

typedef enum
{
	MOTOR_OK,
	MOTOR_NOT_OK,
	MOTOR_NULL_POINTER,
	MOTOR_SPEED_LIMIT_ERROR

} motor_status_t;

/*	section : functions declaration */
std_return_type ecu_motor_init(const motor_t *motor);
std_return_type ecu_motor_move_counter_clock_wise(motor_t *motor);
std_return_type ecu_motor_move_clock_wise(motor_t *motor);
std_return_type ecu_motor_brake(motor_t *motor);
std_return_type ecu_motor_off(motor_t *motor);
std_return_type ecu_motor_change_speed(motor_t *motor, uint8_t _speed);

#endif /* INC_ECU_MOTOR_H_ */
