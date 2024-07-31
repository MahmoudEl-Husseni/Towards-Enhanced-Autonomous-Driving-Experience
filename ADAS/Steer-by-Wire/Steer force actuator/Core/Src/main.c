

/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2024 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes -----------------------------------	-------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

I2C_HandleTypeDef hi2c1;

TIM_HandleTypeDef htim2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);

/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ---- App data types declaration ----	*/

/* ---- Steering Profile ---- */

// 180 degree profile
steering_profile_t SteeringProfile = {.SteeringLockPositionRightDeadLock = 450,
									  .SteeringLockPositionLeftDeadLock = -450};

/*------------------------*/
float degree;
int16_t prev_degree;
int16_t pos;
uint32_t time;
motor_t motor_1;
rotary_encoder_t encoder_1;
encoder_rotation_status_t encoder_1_rot;
int32_t motor_speed_pid;
uint8_t fun_error_ret_val;
uint32_t tick_start;
const uint8_t motor_max_speed = 10;
float output_motor_speed;

struct FRAME_LIST steering;
struct FRAME_LIST wheel;
struct FRAME_LIST autoPilot_steer_angle;

/*---- CAN Bus variable  ----*/
float *CANptr = (float *)&(steering.FRAME.data.total); // tx
uint8_t *CANsteering_profile_flag =
	(uint8_t *)&(autoPilot_steer_angle.FRAME.data.byte6); // CAN Bus receive
uint8_t *CANdriver_autoPilot_flag =
	(uint8_t *)&(autoPilot_steer_angle.FRAME.data.byte7);
int32_t *CAN_carla_to_steering =
	(int32_t *)&(autoPilot_steer_angle.FRAME.data.total);
uint8_t *CAN_wheel_lock_flag = (uint8_t *)&(wheel.FRAME.data.byte5);
float *CAN_wheel_lock_angle = (float *)&(wheel.FRAME.data.total);
/*---- Steering Limits Variables  ----*/

map_range_t steering_to_carla = {.input_min_range = -450, .input_max_range = 450, .output_max_range = 1, .output_min_range = -1};

map_range_t carla_to_steering = {.input_min_range = -1, .input_max_range = 1, .output_max_range = 450, .output_min_range = -450};
map_range_t PID_TO_MOTOR = {.input_min_range = -10000,
							.input_max_range = 10000,
							.output_max_range = motor_max_speed,
							.output_min_range = (motor_max_speed * -1)};
float carla_wheel_postion;
float wheel_lock_value;
float autoPilot_steering_degree;
uint8_t wheel_lock_flag;

/*----- ********************************************** ----- */

float motor_speed_ouput_map, motor_speed_input_map;
int8_t motor_speed_int;
/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)

{
	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */
	// CAN  ID
	steering.FRAME.IDNUM = 0x123;
	steering.FRAME.DLC = 4;
	wheel.FRAME.IDNUM = 0x105;
	wheel.FRAME.DLC = 5;
	autoPilot_steer_angle.FRAME.IDNUM = 0x169;
	autoPilot_steer_angle.FRAME.DLC = 8;
	CAN_START();
	CAN_Add_Filter(&autoPilot_steer_angle, receive);
	CAN_Add_Filter(&steering, transmit);
	//	CAN_Add_Filter(&wheel, receive);

	// transmit the initial value //

	CAN_TX(&steering);
	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();

	/* USER CODE BEGIN 2 */

	/*---- make initialization of Project peripherals   ----	*/
	app_init_function();
	PID(0, pos, &motor_speed_pid);
	tick_start = HAL_GetTick();
	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1)
	{
		CAN_RX();
		if (!(*CANsteering_profile_flag)) // 450  profile
		{
			SteeringProfile.SteeringLockPositionRightDeadLock = 450;
			SteeringProfile.SteeringLockPositionLeftDeadLock = -450;
		}
		else if ((*CANsteering_profile_flag) == 1) //  180  profile
		{
			SteeringProfile.SteeringLockPositionRightDeadLock = 180;
			SteeringProfile.SteeringLockPositionLeftDeadLock = -180;
		}

		steering_to_carla.input_max_range =
			(SteeringProfile.SteeringLockPositionRightDeadLock);
		steering_to_carla.input_min_range =
			(SteeringProfile.SteeringLockPositionLeftDeadLock);
		carla_to_steering.output_max_range =
			(SteeringProfile.SteeringLockPositionRightDeadLock);
		carla_to_steering.output_min_range =
			(SteeringProfile.SteeringLockPositionLeftDeadLock);

		time = HAL_GetTick() - tick_start;

		/* ---- Read TIM register to calculate degree   */
		ecu_rotary_encoder_measure_postion(&encoder_1, &encoder_1_rot, &pos);

		degree = pos * ENCODER_DEGREE_PER_TICK;
		map(degree, &carla_wheel_postion, &steering_to_carla);

		if (time >= 1000)

		{
			tick_start = HAL_GetTick();

			/* ---- Making the LED indicator for running ----- */

			HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
		}

		if (*CANdriver_autoPilot_flag) // Auto Pilot Mode
		{
			auto_pilot_mode();
		}
		else
		{
			driver_mode();
		}

		if (*CAN_wheel_lock_flag)
		{
			map(*CAN_wheel_lock_angle, &wheel_lock_value, &steering_to_carla);
			//	SteeringProfile.SteeringLockPosition = (uint16_t) wheel_lock_value;
			if (wheel_lock_value >= 0)
			{
				SteeringProfile.SteeringLockPositionRightDeadLock =
					(uint16_t)wheel_lock_value;
			}
			else
			{
				SteeringProfile.SteeringLockPositionRightDeadLock =
					(uint16_t)wheel_lock_value;
			}
		}

		// Postive
		if (motor_speed_int > 0)
		{
			/*----- see if the degree are in range ---- */
			in_between(SteeringProfile.SteeringLockPositionLeftDeadLock,
					   SteeringProfile.SteeringLockPositionRightDeadLock, (degree),
					   &wheel_lock_flag);

			if (!wheel_lock_flag)
			{
				motor_1.speed = LOCK_MOTOR_FORCE;
			}
			else
			{
				motor_1.speed = abs(motor_speed_int);
			}

			ecu_motor_move_clock_wise(&motor_1);
		}
		else if (motor_speed_int < 0)
		{

			/* ---- see the degree in the given range  ---- */
			in_between(SteeringProfile.SteeringLockPositionLeftDeadLock,
					   SteeringProfile.SteeringLockPositionRightDeadLock, degree,
					   &wheel_lock_flag);
			if (!wheel_lock_flag)
			{
				motor_1.speed = LOCK_MOTOR_FORCE;
			}
			else
			{
				motor_1.speed = abs(motor_speed_int);
			}
			ecu_motor_move_counter_clock_wise(&motor_1);
		}
		else
		{
			ecu_motor_brake(&motor_1);
		}

		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
	RCC_OscInitTypeDef RCC_OscInitStruct = {0};
	RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

	/** Initializes the RCC Oscillators according to the specified parameters
	 * in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
	RCC_OscInitStruct.HSIState = RCC_HSI_ON;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
	{
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
	{
		Error_Handler();
	}
}

/**
 * @brief CAN Initialization Function
 * @param None
 * @retval None
 */
/**
 * @brief I2C1 Initialization Function
 * @param None
 * @retval None
 //  */

/**
 * @brief TIM2 Initialization Function
 * @param None
 * @retval None
 */

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStruct = {0};
	/* USER CODE BEGIN MX_GPIO_Init_1 */
	/* USER CODE END MX_GPIO_Init_1 */

	/* GPIO Ports Clock Enable */
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(Built_in_LED_GPIO_Port, Built_in_LED_Pin, GPIO_PIN_SET);

	/*Configure GPIO pin : Built_in_LED_Pin */
	GPIO_InitStruct.Pin = Built_in_LED_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(Built_in_LED_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pin : Steering_reset_postion_Pin */
	GPIO_InitStruct.Pin = Steering_reset_postion_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
	GPIO_InitStruct.Pull = GPIO_PULLUP;
	HAL_GPIO_Init(Steering_reset_postion_GPIO_Port, &GPIO_InitStruct);

	/* USER CODE BEGIN MX_GPIO_Init_2 */
	/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
/*
 *
 * 	Function responsible for Ai  to control the vichle
 *
 * */
void auto_pilot_mode()
{
	// autoPilot_steering_degree = (*CAN_carla_to_steering) / 1000.0;
	map(((float)(*CAN_carla_to_steering) / 1000.0), &autoPilot_steering_degree, &carla_to_steering);

	PID(autoPilot_steering_degree, degree, &motor_speed_pid);
	map(motor_speed_pid, &output_motor_speed, &PID_TO_MOTOR);
	motor_speed_input_map = motor_speed_pid;
	motor_speed_int = (int8_t)(ceil(output_motor_speed));
}

/*
 *
 * 	Function responsible for driver to control the vichle
 *
 * */

void driver_mode()
{
	PID(0, degree, &motor_speed_pid);
	motor_speed_input_map = motor_speed_pid;
	map(motor_speed_input_map, &output_motor_speed, &PID_TO_MOTOR);
	motor_speed_int = (int8_t)(ceil(output_motor_speed));

	if (abs(prev_degree - (int16_t)degree) >= 5 || time % 1000 == 0)
	{

		*CANptr = carla_wheel_postion;
		CAN_TX(&steering);
		prev_degree = degree;
	}
}
void app_init_function(void)
{
	/*------ DC Motor  Object Initialize  -------*/
	motor_1.ports[LEFT_PORT] = GPIOA;		 /* !< Left Pins Port Selection  */
	motor_1.ports[RIGHT_PORT] = GPIOA;		 /* !< right Pins Port Selection  */
	motor_1.left_pins[ENABLE] = GPIO_PIN_4;	 /* !< Left enable Pin  Selection  */
	motor_1.left_pins[PWM] = GPIO_PIN_6;	 /* !< Left PWM Pin  Selection  */
	motor_1.right_pins[ENABLE] = GPIO_PIN_5; /* !< Right enable Pin  Selection  */
	motor_1.right_pins[PWM] = GPIO_PIN_7;	 /* !< Right PWM Pin  Selection  */
	motor_1.speed = 20;

	/*-------- Encoder Object Initialize -------*/
	encoder_1.Timer = R_encoder_TIM2;

	/*------ EEPROM Object Initialize	*/

	/*------- Functions Section ------ */

	fun_error_ret_val = ecu_rotary_encoder_intialize(&encoder_1);

	fun_error_ret_val = ecu_motor_init(&motor_1);
}

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	uint32_t MSP_Value = *APP_START_ADDRESS;
	Jump_Ptr Gump_Addr = (Jump_Ptr)(*(APP_START_ADDRESS + 1));
	__disable_irq();
	while (1)
	{
		__set_MSP(MSP_Value);

		// HOST_jump_Addr = (uint32_t *)*pAdd;
	}
	/* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
	/* USER CODE BEGIN 6 */
	/* User can add his own implementation to report the file name and line number,
	 ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
	/* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
