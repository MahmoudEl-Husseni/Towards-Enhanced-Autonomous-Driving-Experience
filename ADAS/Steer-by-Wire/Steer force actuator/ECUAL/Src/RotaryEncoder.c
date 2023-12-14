/*
 * RotaryEncoder.c
 *
 *  Created on: Nov 28, 2023
 *      Author: engmu
 */


#include "RotaryEncoder.h"


volatile int32_t postion ;
volatile int32_t final_postion ;
volatile int32_t prev_final_postion ;


volatile int8_t last_state_signal ;
volatile int8_t current_state_signal ;


const int8_t Encoder_states[] = {
    0, -1, 1, 0,
    1, 0, 0, -1,
    -1, 0, 0, 1,
    0, 1, -1, 0};

std_return_type ecu_rotary_encoder_intialize (const rotary_encoder_t * r_encoder)
{
	/*Error Handling of Null Pointers */
	std_return_type ret = ENCODER_NOT_OK  ;
	if (NULL == r_encoder )
	{
		ret = ENCODER_NOT_OK ;
	}
	else{
			/*		Creating Pin Object to HAL Layer 	*/

		GPIO_InitTypeDef pin_obj_A = {
				.Mode = GPIO_MODE_INPUT,
				.Pull = GPIO_PULLUP,
				.Pin = (r_encoder->ROTARY_PIN_OUTPUTA),

		} ;
		GPIO_InitTypeDef pin_obj_B = {
				.Mode = GPIO_MODE_INPUT,
				.Pull = GPIO_PULLUP,
				.Pin = (r_encoder->ROTARY_PIN_OUTPUTB),
		} ;

		/* Passing Parameters to HAL Layer to Initialize Pin  */
		HAL_GPIO_Init((r_encoder ->ROTARY_PORT_OUTPUTA), &pin_obj_A ) ;
		HAL_GPIO_Init((r_encoder ->ROTARY_PORT_OUTPUTB), &pin_obj_B ) ;



			/*	Read Initial Value of Encoder on initialize */
		uint8_t pina_signal = 	HAL_GPIO_ReadPin( (r_encoder->ROTARY_PORT_OUTPUTA ), (r_encoder ->ROTARY_PIN_OUTPUTA) );
		uint8_t pinb_signal =   HAL_GPIO_ReadPin( (r_encoder->ROTARY_PORT_OUTPUTB ), (r_encoder ->ROTARY_PIN_OUTPUTB) ) ;

		/*take the 2 read in one variable */
		last_state_signal =  ( pina_signal | (pinb_signal <<1));

		/** Make start Position with 0     <-  Problem Here **/
		postion = 0 ;
		final_postion = 0 ;
		prev_final_postion = 0 ;
	    ret = ENCODER_OK ;
	}

return  ret ;
}

std_return_type ecu_rotary_encoder_measure_postion(const rotary_encoder_t *r_encoder ,
												   encoder_rotation_status_t *encoder_rotation  )
{

	std_return_type ret = ENCODER_NOT_OK  ;

	if (NULL == r_encoder || NULL == encoder_rotation   )
	{
		ret = ENCODER_NOT_OK ;
	}
	else{


		uint8_t pina_signal = 	HAL_GPIO_ReadPin( (r_encoder->ROTARY_PORT_OUTPUTA ), (r_encoder ->ROTARY_PIN_OUTPUTA) );
		uint8_t pinb_signal =   HAL_GPIO_ReadPin( (r_encoder->ROTARY_PORT_OUTPUTB ), (r_encoder ->ROTARY_PIN_OUTPUTB) ) ;

		/*take the 2 read in one variable */
		current_state_signal =  ( pina_signal | pinb_signal << 1);


			prev_final_postion = final_postion ;

		if (current_state_signal != last_state_signal )
		{

			postion += Encoder_states[(current_state_signal | (last_state_signal<<2) )] ;



			last_state_signal = current_state_signal ;

			if (current_state_signal ==  3 )
			{
				final_postion = postion >> 2 ;
			}


		}
		/*	Know the Type of Rotation 	*/


		if ( final_postion  >  prev_final_postion )
		{
			*encoder_rotation  = ENCODER_CLOCK_WISE ;
		}
		else if ( final_postion < prev_final_postion  )
		{
			*encoder_rotation  = ENCODER_ANTI_CLOCK_WISE ;
		}
		else {

			*encoder_rotation  = ENCODER_DOESNOT_MOVE ;

		}


		ret = ENCODER_OK ;


	}
	return  ret ;

}
std_return_type ecu_rotary_encoder_return_postion_value (int32_t *postion_value)
{
	std_return_type ret = ENCODER_NOT_OK  ;

	if (NULL == postion_value   )
	{
		ret = ENCODER_NOT_OK ;
	}
	else{


		*postion_value = final_postion ;

		ret = ENCODER_OK ;
	}
	return  ret ;
}

