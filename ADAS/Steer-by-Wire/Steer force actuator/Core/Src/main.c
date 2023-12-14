/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "SERVO.h"
#include "RotaryEncoder.h"
#include "ecu_motor.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define SERVO_Motor1   0
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

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

  /*rotary_encoder_t rotary_1  = {
	.ROTARY_PORT_OUTPUTA = GPIOA ,
	.ROTARY_PORT_OUTPUTB = GPIOA ,
	.ROTARY_PIN_OUTPUTA  = GPIO_PIN_11 ,
	.ROTARY_PIN_OUTPUTB  = GPIO_PIN_12
  };
*/
  motor_t motor1 = {
		  .ports[LEFT_PORT] = GPIOA ,
		  .ports[RIGHT_PORT] = GPIOA ,
		  .left_pins[PWM] = GPIO_PIN_0 ,
		  .right_pins[PWM] = GPIO_PIN_6 ,
		  .left_pins[ENABLE] = GPIO_PIN_3,
		  .right_pins[ENABLE] = GPIO_PIN_4
  };

 // encoder_rotation_status_t encoder_rotation = ENCODER_DOESNOT_MOVE ;
  //int32_t pos  = 0 ;
  //int32_t last_pos = 0 ;
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  /* USER CODE BEGIN 2 */

	//SERVO_Init(SERVO_Motor1);
	//ecu_rotary_encoder_intialize(&rotary_1) ;
	ecu_motor_init(&motor1, 50) ;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1)
	{

	/*Read the value of position */


	/*Measure Rotation and update position  */
		/*
	ecu_rotary_encoder_measure_postion(&rotary_1, &encoder_rotation) ;

	ecu_rotary_encoder_return_postion_value(&pos ) ;

	 if (last_pos != pos )
	{
		SERVO_MoveTo(SERVO_Motor1, ( (abs(pos)* 18)  %180)  );
	}
	 HAL_GetTick() ;
	 last_pos = pos ;
	//HAL_Delay(200);


		for ( int16_t  i = 0 ; i < 180 ; i+= 15  )
		{
	    	SERVO_MoveTo(SERVO_Motor1, i);
	    	HAL_Delay(500);

		}
		for ( int16_t  i = 180 ; i >0 ; i-= 15  )
		{
	    	SERVO_MoveTo(SERVO_Motor1, i);
	    	HAL_Delay(500);

		}

    	SERVO_MoveTo(SERVO_Motor1, 0);
    	HAL_Delay(1000);
    	SERVO_MoveTo(SERVO_Motor1, 45);
    	HAL_Delay(500);
    	SERVO_MoveTo(SERVO_Motor1, 90);
    	HAL_Delay(500);
    	SERVO_MoveTo(SERVO_Motor1, 135);
    	HAL_Delay(500);
    	SERVO_MoveTo(SERVO_Motor1, 180);
    	HAL_Delay(500);
    	SERVO_MoveTo(SERVO_Motor1, 90);
    	HAL_Delay(1000);*/
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
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
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
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOD_CLK_ENABLE();

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1)
	{
	}
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
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
