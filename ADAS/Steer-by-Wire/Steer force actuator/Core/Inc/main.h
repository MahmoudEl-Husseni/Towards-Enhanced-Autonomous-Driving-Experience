/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "../../ECUAL/_includes_.h"

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */
/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */
void  app_init_function (void ) ;
void auto_pilot_mode () ;
void driver_mode () ;

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define Built_in_LED_Pin GPIO_PIN_13
#define Built_in_LED_GPIO_Port GPIOC
#define Steering_reset_postion_Pin GPIO_PIN_5
#define Steering_reset_postion_GPIO_Port GPIOA
#define APP_START_ADDRESS ((uint32_t * )0x8000000)

/* USER CODE BEGIN Private defines */
#define ENCODER_DEGREE_PER_TICK  (float)0.6
#define FLAG_BIT_CLOCK_WISE_DIRECTION 0
#define FLAG_BIT_ANTI_CLOCK_WISE_DIRECTION 1

#define LOCK_MOTOR_FORCE 50
 typedef struct {
	 int16_t SteeringLockPositionRightDeadLock ;
	 int16_t SteeringLockPositionLeftDeadLock ;
 }steering_profile_t ;
 typedef void (*Jump_Ptr)(void);
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
