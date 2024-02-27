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
1- EEPROM
  - Select I2C bus From I2C1 or I2C2
  - Select EEPROM Type ( AT24C01 - AT24C02 -AT24C04 -AT24C08-AT24C016)
For Example in [eeprom_config.h](ECUAL/EEPROM/eeprom_config.h) File  :

'''CPP
#inlcude <stdio.h>
int main()
{
return  0 ; 
}
'''

>[!CAUTION]
>Make Sure that STM Cube Mx Copy the Necessary Files in the project


