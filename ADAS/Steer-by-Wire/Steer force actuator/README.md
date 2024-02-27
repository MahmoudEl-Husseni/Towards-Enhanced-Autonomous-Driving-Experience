# Steer Force Actuator 
> This File are Compatible with CMSIS API Function Genrated by STMCube Mx ide  written f

## Description 
Steer Force Actuator System is Steer-by-wire Sub system responsible for taking the order from Driver and Interface with the vichele  

## System Flow Chart 
![Steer-by-wire](https://github.com/Muhammad-Osama-9/Towards-Enhanced-Autonomous-Driving-Experience/assets/112892754/ab71835a-70e0-4510-adff-927ab3954d5f)


### ***System Component***

* [STM32F103C8t6](https://www.st.com/resource/en/datasheet/stm32f103c8.pdf) 
* Dc Motor 775
* HW-039 Motor Driver (High Amp)
* Rotary Encoder (KY-040)
* [EEPROM AT24C04](https://ww1.microchip.com/downloads/en/devicedoc/doc0180.pdf)




## ***Project Pin Diagram in MX***
![1122](https://github.com/Muhammad-Osama-9/Towards-Enhanced-Autonomous-Driving-Experience/assets/112892754/bc2d8295-c665-447c-898c-c8e7c57de1a3)

## Code Section Notes 
* EEPROM
    - Select I2C bus From I2C1 or I2C2
    - Select EEPROM Type ( AT24C01 - AT24C02 -AT24C04 -AT24C08-AT24C016)
For Example in [eeprom_config.h](ECUAL/EEPROM/eeprom_config.h) File  :

```cpp
#ifndef EEPROM_EEPROM_CONFIG_H_
#define EEPROM_EEPROM_CONFIG_H_

/*	 Config Macros for AT24Cxx EEPROM 	*/

//#define _24C01
//#define _24C02
#define _24C04
//#define _24C08
//#define _24C016

 /* Select I2C Bus selection  */

#define I2C_1
//#define I2C_2


#endif /* EEPROM_EEPROM_CONFIG_H_ */
```

***this Seelect 24C04 eeprom and I2C1 Bus***

### Pins Diagram 
| I2CBus | SDA |  SCL |
| :---:   | :---:  | :---:   |
| I2C1  | PB7  |  PB6  |
|I2C2   | PB11 | PB0  |

* Rotary Encoder 
    - Select  Encoder From 2 Modes 
        - Custom Implementation functions 
        - Timer in Encoder Mode
for Examlple in [rotary_encoder_CONF.h](ECUAL/Rotary_Encoder/rotary_encoder_CONF.h) File  :

### Pins Diagram 
| Tim    | CH1 |  CH2 |
| :---:  | :---: | :---:  |
| TIM1   | PA8  |  PA9  |
| TIM2   | PA0 | PA1  |
| TIM3   | PA6 | PA7  |


```cpp
#ifndef ROTARY_ENCODER_ROTARY_ENCODER_CONF_H_
 #define ROTARY_ENCODER_ROTARY_ENCODER_CONF_H_
 #define TIM_ENCODER   // Select from 2 Encoder Modes
#endif /* ROTARY_ENCODER_ROTARY_ENCODER_CONF_H_ */
 ```
### Dc Motor 

   -  choose only the pins you wants to Connect int motor_t  
   -  timir_init () ; Configure the timer and channel crossponding for this pins with no MAPPING

### ***Timer Channel With No Mapping***

| Tim    | CH1 |  CH2 | CH3 | CH4 |
| :---:  | :---: | :---:  | | :---: | :---:  |
| TIM1   | PA8  |  PA9  | PA10 | PA11 |
| TIM2   | PA0 | PA1  | PA2 | PA3 |
| TIM3   | PA6 | PA7  | PB0 | PB1 |
| TIM4   | PB8 | PB9  | PB10 | PB11 |

>[!NOTE]
> Dont Use TIM4 bec it disbale the I2c Bus

### Example of motor_t initializtaion 
```c
motor_t motor_1 = {
		.left_pins[ENABLE] = GPIO_PIN_4 ,    
		.left_pins[PWM] = GPIO_PIN_6 , 
		.right_pins[ENABLE] = GPIO_PIN_5 ,
		.right_pins[PWM] = GPIO_PIN_7 , 
		speed = 50 
};
```
> This Only Parameters you Need to Defines

>[!CAUTION]
>Make Sure that STM Cube Mx Copy the Necessary Files in the project


