/*
 * ecu_motor.c
 *
 *  Created on: Dec 8, 2023
 *      Author: engmu
 */

#include "ecu_motor.h"


	/*	Time Handler Object	*/
 TIM_HandleTypeDef htim = {0};

const uint16_t porta_timer_channels [] =
{
		TIM_CHANNEL_1 , TIM_CHANNEL_2 , TIM_CHANNEL_3,TIM_CHANNEL_4,  // TIM2 Pins  0 , 1 ,2 , 3
		RESERVED , RESERVED ,   									  // RESERVED PINS 	4 , 5
		TIM_CHANNEL_1 , TIM_CHANNEL_2 ,								  // TIM3 Pins 6 ,7
		TIM_CHANNEL_1 , TIM_CHANNEL_2 , TIM_CHANNEL_3,TIM_CHANNEL_4   // TIM1 Pins 8,9,10,11

};

const uint16_t portb_timer_channels [] =
{
		 TIM_CHANNEL_3,TIM_CHANNEL_4,  						   		   // TIM3 Pins  0 , 1
		 RESERVED , RESERVED ,  RESERVED , RESERVED ,  		           // RESERVED PINS 	2,3,4,5
		 TIM_CHANNEL_1 , TIM_CHANNEL_2 , TIM_CHANNEL_3,TIM_CHANNEL_4   // TIM4 Pins 6,7,8,9

};

static int16_t  log2fun (uint32_t num  )
{
	uint16_t res = 0 ;
	while (num )
	{
		res++ ;
		num/=2 ;
	}
	res -- ;

	return res ;
}

static void timers_init ( motor_t *motor )
{
	/*	declaration  variables   */

	TIM_ClockConfigTypeDef  sClockSourceConfig = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};
    TIM_OC_InitTypeDef	      sConfigOC = {0};

    uint32_t PSC_Value = 0;
    uint32_t ARR_Value = 0;
    uint32_t tim_channel = 0  ;
	//DWT_Delay_Init();

    /****************************************
     * 			LEFT PIN TIM CONFIG			*
     ****************************************/

	/*Enable RCC Port Clocks and Timers */

	if (motor->ports[LEFT_PORT] == GPIOA )
	{
		/* Enable RCC Clock */

		__HAL_RCC_GPIOA_CLK_ENABLE() ;

		/*	Enable Timer Corresponding  to its pin 	*/

		if (   motor->left_pins[PWM] == GPIO_PIN_0
			|| motor->left_pins[PWM] == GPIO_PIN_1
			|| motor->left_pins[PWM] == GPIO_PIN_2
			|| motor->left_pins[PWM] == GPIO_PIN_3 )
		{
				   __HAL_RCC_TIM2_CLK_ENABLE();
				   htim.Instance = TIM2 ;
				   motor->left_TIM_Instance = TIM2 ;

		}else if (  motor->left_pins[PWM]==GPIO_PIN_6
				 || motor->left_pins[PWM]==GPIO_PIN_7 )
			{
					__HAL_RCC_TIM3_CLK_ENABLE();

					htim.Instance = TIM3 ;
					motor->left_TIM_Instance = TIM3 ;


		}else if ( motor->left_pins[PWM] == GPIO_PIN_8
				|| motor->left_pins[PWM] == GPIO_PIN_9
				|| motor->left_pins[PWM] == GPIO_PIN_10
				|| motor->left_pins[PWM] == GPIO_PIN_11 )

		 	{
					__HAL_RCC_TIM1_CLK_ENABLE();

					 htim.Instance = TIM1 ;
					 motor->left_TIM_Instance = TIM1 ;

			}


		tim_channel = porta_timer_channels[log2fun(motor->left_pins[PWM])] ;

	}
	else if (motor->ports[LEFT_PORT] == GPIOB)
	{
		/* Enable RCC Clock */
		__HAL_RCC_GPIOB_CLK_ENABLE() ;

		/*	Enable Timer Corresponding  to its pin 	*/

		 if (motor->left_pins[PWM] == GPIO_PIN_6
		  || motor->left_pins[PWM] == GPIO_PIN_7
		  || motor->left_pins[PWM] == GPIO_PIN_8
		  || motor->left_pins[PWM] == GPIO_PIN_9 )
		 {
			 __HAL_RCC_TIM4_CLK_ENABLE() ;

			 	htim.Instance = TIM4 ;
			   motor->left_TIM_Instance = TIM4 ;

		 }
		 else if (	 motor->left_pins[PWM] == GPIO_PIN_0
				  || motor->left_pins[PWM] == GPIO_PIN_1  )
		 {
			 __HAL_RCC_TIM3_CLK_ENABLE() ;
			  htim.Instance = TIM3 ;
			  motor->left_TIM_Instance = TIM3 ;

		 }

		 tim_channel = portb_timer_channels[log2fun(motor->left_pins[PWM])] ;

	}

	else {/*Do Nothing*/}
	//============================================================================================//


	/*	Determine CCRX According to Channel type (1,2,3,4) in left Pins */

	switch (tim_channel)
	{
	case TIM_CHANNEL_1 :
		motor->left_TIM_CCRx=&(motor->left_TIM_Instance->CCR1) ;
		break ;

	case TIM_CHANNEL_2 :
		motor->left_TIM_CCRx=&(motor->left_TIM_Instance->CCR2) ;
		break ;

	case TIM_CHANNEL_3 :
		motor->left_TIM_CCRx=&(motor->left_TIM_Instance->CCR3) ;
		break ;

	case TIM_CHANNEL_4 :
		motor->left_TIM_CCRx=&(motor->left_TIM_Instance->CCR4) ;
		break ;

	}


	PSC_Value = (uint32_t) (CPU_FREQ / 3276800.0);
	ARR_Value = (uint32_t) ((CPU_FREQ / (PWM_FREQ*(PSC_Value+1.0)))-1.0);



	/*	put values in htim object */
	htim.Init.Prescaler = PSC_Value ;
	htim.Init.Period = ARR_Value 	;
	htim.Init.CounterMode = TIM_COUNTERMODE_UP ;
	htim.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
	htim.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
	HAL_TIM_Base_Init(&(htim));

	sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
	HAL_TIM_ConfigClockSource(&htim, &sClockSourceConfig);
	HAL_TIM_PWM_Init(&(htim));

	sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
	sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
	HAL_TIMEx_MasterConfigSynchronization(&htim, &sMasterConfig);

	sConfigOC.OCMode = TIM_OCMODE_PWM1;
	sConfigOC.Pulse = 0;
	sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
	sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
	HAL_TIM_PWM_ConfigChannel(&htim, &sConfigOC, tim_channel);

	/* -----	Start PWM	------*/

	HAL_TIM_PWM_Start(&(htim),tim_channel );





//==============================================================================================================//



/****************************************
 * 			RIGHT PIN TIM CONFIG		*
****************************************/


	if (motor->ports[RIGHT_PORT] == GPIOA)  // -> Port A <- //
	{

		__HAL_RCC_GPIOA_CLK_ENABLE() ;
		/* Know Each Timer according to pin to Enable */

				if (   motor->right_pins[PWM] == GPIO_PIN_0
					|| motor->right_pins[PWM] == GPIO_PIN_1
					|| motor->right_pins[PWM] == GPIO_PIN_2
					|| motor->right_pins[PWM] == GPIO_PIN_3 )
				{
						   __HAL_RCC_TIM2_CLK_ENABLE();

						  htim.Instance = TIM2 ;
						  motor->right_TIM_Instance = TIM2 ;


				}else if ( motor->right_pins[PWM] == GPIO_PIN_6
						|| motor->right_pins[PWM] == GPIO_PIN_7 )
					{
							__HAL_RCC_TIM3_CLK_ENABLE();

							  htim.Instance = TIM3 ;
							  motor->right_TIM_Instance = TIM3 ;


				}else if ( motor->right_pins[PWM] == GPIO_PIN_8
						|| motor->right_pins[PWM] == GPIO_PIN_9
						|| motor->right_pins[PWM] == GPIO_PIN_10
						|| motor->right_pins[PWM] == GPIO_PIN_11 )

				 	{
							__HAL_RCC_TIM1_CLK_ENABLE();

							  htim.Instance = TIM1 ;
							  motor->right_TIM_Instance = TIM1 ;
					}

		tim_channel = porta_timer_channels[log2fun(motor->right_pins[PWM])] ;


	}
	else if (motor->ports[RIGHT_PORT] == GPIOB)
	{
		/* Enable RCC Clock */
				__HAL_RCC_GPIOB_CLK_ENABLE() ;

				/*	Enable Timer Corresponding  to its pin 	*/

				 if (motor->right_pins[PWM] == GPIO_PIN_6
				  || motor->right_pins[PWM] == GPIO_PIN_7
				  || motor->right_pins[PWM] == GPIO_PIN_8
				  || motor->right_pins[PWM] == GPIO_PIN_9 )
				 {
					 __HAL_RCC_TIM4_CLK_ENABLE() ;

					  htim.Instance = TIM4 ;
					  motor->right_TIM_Instance = TIM4 ;


				 }
				 else if (	 motor->right_pins[PWM] == GPIO_PIN_0
						  || motor->right_pins[PWM] == GPIO_PIN_1  )
				 {
					 __HAL_RCC_TIM3_CLK_ENABLE() ;

					 htim.Instance = TIM3 ;
					 motor->right_TIM_Instance = TIM3 ;

				 }

				 /*	Determine the channel type from its port and pin  */

		tim_channel = portb_timer_channels[log2fun(motor->right_pins[PWM])] ;


	}
	else {/*Do Nothing */}

//============================================================================================//

	/*	Determine CCRX According to Channel type (1,2,3,4) in left Pins */
	switch (tim_channel)
		{
		case TIM_CHANNEL_1 :
			motor->right_TIM_CCRx=&(motor->right_TIM_Instance->CCR1) ;
			break ;

		case TIM_CHANNEL_2 :
			motor->right_TIM_CCRx=&(motor->right_TIM_Instance->CCR2) ;
			break ;

		case TIM_CHANNEL_3 :
			motor->right_TIM_CCRx=&(motor->right_TIM_Instance->CCR3) ;
			break ;

		case TIM_CHANNEL_4 :
			motor->right_TIM_CCRx=&(motor->right_TIM_Instance->CCR4) ;
			break ;

		}

		/*Calculate Prescaler and ARR*/

	PSC_Value = (uint32_t) (CPU_FREQ / 3276800.0);
	ARR_Value = (uint32_t) ((CPU_FREQ / (PWM_FREQ*(PSC_Value+1.0)))-1.0);



	/*	put values in htim object */

	htim.Init.Prescaler = PSC_Value ;
	htim.Init.Period = ARR_Value 	;
	htim.Init.CounterMode = TIM_COUNTERMODE_UP ;
	htim.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
	htim.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
	HAL_TIM_Base_Init(&(htim));

	sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
	HAL_TIM_ConfigClockSource(&htim, &sClockSourceConfig);
	HAL_TIM_PWM_Init(&(htim));

	sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
	sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
	HAL_TIMEx_MasterConfigSynchronization(&htim, &sMasterConfig);

	sConfigOC.OCMode = TIM_OCMODE_PWM1;
	sConfigOC.Pulse = 0;
	sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
	sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
	HAL_TIM_PWM_ConfigChannel(&htim, &sConfigOC, tim_channel);

		/*---- Start PWM Channel -----*/

	HAL_TIM_PWM_Start(&(htim),tim_channel );


}



std_return_type ecu_motor_init 	 (const motor_t *motor  , uint8_t speed)
{




	uint8_t ret = MOTOR_NOT_OK;
	if(NULL == motor )
	{
		ret = MOTOR_NOT_OK ;
	}
	else {

		/*	Limit speed to be 0 to 100 	*/
		if (speed >100)
			{
				speed = 100 ;
			}
			/* Init Timers used in PWM */

		float duty_cycle = 1-(speed/100.0) ;

		timers_init(motor) ;

		/*Make GPIO pin Objects */
			/* Right Pins 	*/
		GPIO_InitTypeDef right_enable_pin_obj = {
				.Pin = motor->right_pins[ENABLE],
				.Mode = GPIO_MODE_OUTPUT_PP,
				.Speed = GPIO_SPEED_FREQ_LOW
		};
		GPIO_InitTypeDef right_PWM_pin_obj = {
				.Pin = motor->right_pins[PWM],
				.Mode = GPIO_MODE_AF_PP,
				.Speed = GPIO_SPEED_FREQ_LOW
		};

			/* left  Pins 	*/
		GPIO_InitTypeDef left_enable_pin_obj = {
			.Pin = motor->left_pins[ENABLE],
			.Mode = GPIO_MODE_OUTPUT_PP,
			.Speed = GPIO_SPEED_FREQ_LOW
		};
		GPIO_InitTypeDef left_PWM_pin_obj = {
			.Pin = motor->left_pins[PWM],
			.Mode = GPIO_MODE_AF_PP,
			.Speed = GPIO_SPEED_FREQ_LOW
		};

		/* init left pins  in HAL layer */
		HAL_GPIO_Init(motor->ports[LEFT_PORT], &left_enable_pin_obj ) ;
		HAL_GPIO_Init(motor->ports[LEFT_PORT], &left_PWM_pin_obj ) ;

		/* init right pins  in HAL layer */
		HAL_GPIO_Init(motor->ports[RIGHT_PORT], &right_enable_pin_obj ) ;
		HAL_GPIO_Init(motor->ports[RIGHT_PORT], &right_PWM_pin_obj   ) ;

		/* Mapping speed from 1 -> 100 	*/

		HAL_GPIO_WritePin(motor->ports[LEFT_PORT], motor->left_pins[ENABLE], GPIO_PIN_RESET) ;
		HAL_GPIO_WritePin(motor->ports[RIGHT_PORT], motor->right_pins[ENABLE], GPIO_PIN_RESET) ;




		*(motor->left_TIM_CCRx) =((uint32_t)  (duty_cycle * htim.Init.Period))   ;
		*(motor->right_TIM_CCRx) =((uint32_t)  (duty_cycle  * htim.Init.Period))  ;

	//	HAL_GPIO_WritePin(motor->ports[LEFT_PORT], motor->left_pins[ENABLE], GPIO_PIN_RESET) ;
		ret = MOTOR_OK ;
	}
	return ret ;
}
std_return_type ecu_motor_move_left  (const motor_t *motor)
{
	uint8_t ret = MOTOR_NOT_OK;
	if(NULL == motor )
	{
		ret = MOTOR_NOT_OK ;
	}
	else {

	}
	return ret ;

}
std_return_type ecu_motor_move_right (const motor_t *motor)
{
	uint8_t ret = MOTOR_NOT_OK;
	if(NULL == motor )
	{
		ret = MOTOR_NOT_OK ;
	}
	else {

	}
	return ret ;
}
std_return_type ecu_motor_brake	 (const motor_t *motor)
{
	uint8_t ret = MOTOR_NOT_OK;
	if(NULL == motor )
	{
		ret = MOTOR_NOT_OK ;
	}
	else {

	}
	return ret ;
}
std_return_type ecu_motor_off 	(const motor_t *motor)
{
	uint8_t ret = MOTOR_NOT_OK;
	if(NULL == motor )
	{
		ret = MOTOR_NOT_OK ;
	}
	else {

	}
	return ret ;


}

std_return_type ecu_motor_change_speed  (const motor_t *motor , uint8_t speed)
{
	uint8_t ret = MOTOR_NOT_OK;
	if(NULL == motor )
	{
		ret = MOTOR_NOT_OK ;
	}
	else {

	}
	return ret ;

}


