/*
 * RotaryEncoder.h
 *
 *  Created on: Nov 28, 2023
 *      Author: engmu
 */

#ifndef INC_ROTARYENCODER_H_
#define INC_ROTARYENCODER_H_

/* Section  : include 	*/

#include "stm32f1xx_hal.h"



/* Section  : User Defined Data types  	*/

typedef enum {
	ENCODER_OK  ,
	ENCODER_NOT_OK
}encoder_status_t;

typedef enum {
	ENCODER_CLOCK_WISE ,
	ENCODER_ANTI_CLOCK_WISE ,
	ENCODER_DOESNOT_MOVE

}encoder_rotation_status_t;

typedef struct {
	GPIO_TypeDef * ROTARY_PORT_OUTPUTA  ;
	GPIO_TypeDef * ROTARY_PORT_OUTPUTB  ;
	uint16_t       ROTARY_PIN_OUTPUTA   ;
	uint16_t       ROTARY_PIN_OUTPUTB   ;


}rotary_encoder_t;


typedef uint8_t std_return_type ;
/* Section  : Functions Declaration  	*/

std_return_type ecu_rotary_encoder_intialize (const rotary_encoder_t * r_encoder) ;

std_return_type ecu_rotary_encoder_measure_postion(const rotary_encoder_t *r_encoder ,
												   encoder_rotation_status_t *encoder_rotation  );

std_return_type ecu_rotary_encoder_return_postion_value (int32_t *postion_value);
#endif /* INC_ROTARYENCODER_H_ */
