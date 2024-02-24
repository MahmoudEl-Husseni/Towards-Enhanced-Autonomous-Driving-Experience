/*
 * rotary_encoder.h
 *
 *  	@Date   : Feb 19, 2024
 *      @Author : Muhammad Osama ELaraby
 *      @breif  : Driver for Incremental  Rotary Encoder (KY040) with necessary function
 *      with Custom implementation functions and timer Module
 *      with option to use ONLY one by
 *      define TIM_ENCODER  MACRO -> to use timer or
 *      not define TIM_ENCODER  MACRO -> to use Custom implementation
 */

#ifndef ROTARY_ENCODER_ROTARY_ENCODER_H_
#define ROTARY_ENCODER_ROTARY_ENCODER_H_
/* ------Section :Includes ------ */

#include "../Util/util.h"

/* ------ Section  : User Defined Data types ------- */

 #define TIM_ENCODER   // Select from 2 Encoder Modes



#ifndef TIM_ENCODER

/* ------ Section  : User Defined Data types ------- */

typedef struct {
	GPIO_TypeDef * ROTARY_PORT_OUTPUTA  ;
	GPIO_TypeDef * ROTARY_PORT_OUTPUTB  ;
	uint16_t       ROTARY_PIN_OUTPUTA   ;
	uint16_t       ROTARY_PIN_OUTPUTB   ;


}rotary_encoder_t;


#endif /* NOT_DEF_TIM_ENCODE */

#ifdef TIM_ENCODER
/* ------ Section  : User Defined Data types ------- */
typedef struct {
	TIM_HandleTypeDef htim ;
}rotary_encoder_t;

static void encoder_htim_par (rotary_encoder_t * r_encoder) ;
#endif/* DEF_TIM_ENCODE */

/* ------ Section  : User Defined Data types ------- */

typedef enum {
	ENCODER_OK  ,
	ENCODER_NOT_OK,
	ENCODER_NULL_POINTER,
	ENCODER_INIT_ERROR,
	ENCODER_MASTER_INIT_ERROR,
	ENCODER_START_ERROR
}encoder_status_t;

typedef enum {
	ENCODER_CLOCK_WISE ,
	ENCODER_ANTI_CLOCK_WISE ,
	ENCODER_DOESNOT_MOVE

}encoder_rotation_status_t;


/*------- Section  : Functions Declaration -------- */
std_return_type ecu_rotary_encoder_intialize (const rotary_encoder_t * r_encoder) ;

std_return_type ecu_rotary_encoder_measure_postion(const rotary_encoder_t *r_encoder ,
												   encoder_rotation_status_t *encoder_rotation  );

std_return_type ecu_rotary_encoder_return_postion_value (int32_t *postion_value);



#endif /* ROTARY_ENCODER_ROTARY_ENCODER_H_ */
