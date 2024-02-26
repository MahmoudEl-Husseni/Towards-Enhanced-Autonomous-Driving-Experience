/*
 * rotary_encoder.c
 *
 *  Created on: Feb 19, 2024
 *      Author: engmu
 */

#include "rotary_encoder.h"


//volatile int test __attribute__((section(".attia_array")))  = 5 ;
int32_t  postion  ;

volatile int32_t final_postion ;
volatile int32_t prev_final_postion ;


volatile int8_t last_state_signal ;
volatile int8_t current_state_signal ;


#ifndef TIM_ENCODER
/*		detect all encoder states   */
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
#endif /* NOT_DEF_TIM_ENCODE */
/*	---- use  encoder in timer module ----- */

#ifdef TIM_ENCODER


std_return_type ecu_rotary_encoder_intialize ( const rotary_encoder_t * r_encoder) {

	/*Error Handling of Null Pointers */
	std_return_type ret = ENCODER_NOT_OK  ;
	if (NULL == r_encoder )
	{
		ret = ENCODER_NULL_POINTER ;
	}
	else{
		TIM_HandleTypeDef htim ;
		TIM_Encoder_InitTypeDef sConf = {0};
		TIM_MasterConfigTypeDef sMaster = {0} ;
		if (r_encoder->Timer == R_encoder_TIM1)
		{
			htim.Instance = TIM1 ;
		}
		else if (r_encoder->Timer == R_encoder_TIM2)
		{
			htim.Instance = TIM2 ;
		}
		else if (r_encoder->Timer == R_encoder_TIM3)
		{
			htim.Instance = TIM3 ;
		}else {
			ret = ENCODER_TIM_SELECTION_ERROR ;
		}

		encoder_htim_par (&htim , &sConf ) ;


		 if (HAL_TIM_Encoder_Init(&(htim), &sConf) != HAL_OK)
		  {
			 ret = ENCODER_INIT_ERROR ;
		  }

		 sMaster.MasterOutputTrigger = TIM_TRGO_RESET ;
		 sMaster.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE ;

		 if (HAL_TIMEx_MasterConfigSynchronization(&(htim), &sMaster) != HAL_OK)
		   {
			ret = ENCODER_MASTER_INIT_ERROR ;
		   }

	if (HAL_TIM_Encoder_Start(&(htim), TIM_CHANNEL_ALL) != HAL_OK)
	{
		ret = ENCODER_START_ERROR ;
	}
	ret = ENCODER_OK ;
}//else
	return ret ;
}

static void encoder_htim_par (TIM_HandleTypeDef * htim ,
							  TIM_Encoder_InitTypeDef *sConf )
{


	htim->Init.Prescaler = 0 ;

	htim->Init.CounterMode=TIM_COUNTERMODE_UP ;

	htim->Init.Period = 0xFFFF ; // (65535)

	htim->Init.ClockDivision = TIM_CLOCKDIVISION_DIV1 ;

	htim->Init.RepetitionCounter = 0 ;

	htim->Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;

	sConf->EncoderMode = TIM_ENCODERMODE_TI12;

	sConf->IC1Polarity = TIM_ICPOLARITY_RISING ;
		// Input Filter to channel //
	sConf->IC1Filter = 10 ;

	sConf->IC1Selection= TIM_ICSELECTION_DIRECTTI;

	sConf->IC1Prescaler = TIM_ICPSC_DIV1;

	sConf->IC2Polarity = TIM_ICPOLARITY_RISING ;
		// Input Filter to channel //
	sConf->IC2Filter = 10 ;

	sConf->IC2Selection= TIM_ICSELECTION_DIRECTTI;

	sConf->IC2Prescaler = TIM_ICPSC_DIV1;

}

std_return_type ecu_rotary_encoder_measure_postion(const rotary_encoder_t *r_encoder ,
												   encoder_rotation_status_t *encoder_rotation  , int16_t *postion )
{
	std_return_type ret = ENCODER_NOT_OK ;
	if (NULL == r_encoder ||NULL == encoder_rotation ){
			ret = ENCODER_NULL_POINTER ;
	}
	else {
		/*	---- Access the corresponding Timer to see the encoder Value ------	*/
		if (r_encoder->Timer == R_encoder_TIM1)
				{
						*postion = ((TIM1->CNT)>>2) ;
				}
				else if (r_encoder->Timer == R_encoder_TIM2)
				{
					*postion = ((TIM2->CNT)>>2) ;
				}
				else if (r_encoder->Timer == R_encoder_TIM3)
				{
					*postion = ((TIM3->CNT)>>2) ;
				}else {
					ret = ENCODER_TIM_SELECTION_ERROR ;
				}
			if (ret !=ENCODER_TIM_SELECTION_ERROR )
			{
				ret = ENCODER_OK ;
			}

	}
	return ret  ;
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

#endif /* DEF_TIM_ENCODE */
