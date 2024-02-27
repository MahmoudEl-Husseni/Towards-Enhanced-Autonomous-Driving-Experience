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

*Rotary Encoder 
    - Select  Encoder From 2 Modes 
        - Custom Implementation functions 
        - Timer in Encoder Mode
for Examlple in [rotary_encoder_CONF.h](ECUAL/Rotary_Encoder/rotary_encoder_CONF.h) File  :
        
```cpp
#ifndef ROTARY_ENCODER_ROTARY_ENCODER_CONF_H_
 #define ROTARY_ENCODER_ROTARY_ENCODER_CONF_H_
 #define TIM_ENCODER   // Select from 2 Encoder Modes
#endif /* ROTARY_ENCODER_ROTARY_ENCODER_CONF_H_ */
 ```

>[!CAUTION]
>Make Sure that STM Cube Mx Copy the Necessary Files in the project


