/*
 * eeprom.c
 *
 *  Created on: Feb 19, 2024
 *      Author: engmu
 */

#include "eeprom.h"

/* ----- I2C Initialization Variable  ---- */
I2C_HandleTypeDef hi2c;


std_return_type eeprom_init(const eeprom_t *eeprom_instace){
	std_return_type ret = EEPROM_OK;
	if (NULL == eeprom_instace)
	{
		ret = EEPROM_NULL_POINTER ;
	}else {
		uint8_t eeprom_address  = EEPROM_ADDRESS(eeprom_instace->address.address_pins_total);

		/*HAL status Variable */
		uint8_t hal_stat ;

		/* ------ Select i2c Bus ---- */
#if  defined (I2C_1)

		  hi2c.Instance = I2C1;
		  hi2c.Init.ClockSpeed = 100000;
		  hi2c.Init.DutyCycle = I2C_DUTYCYCLE_2;
		  hi2c.Init.OwnAddress1 = 0;
		  hi2c.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
		  hi2c.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
		  hi2c.Init.OwnAddress2 = 0;
		  hi2c.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
		  hi2c.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
		  if (HAL_I2C_Init(&hi2c) != HAL_OK)
		  {
		    ret = EEPROM_INIT_ERROR ;
		  }
#endif// I2C_1

#if  defined (I2C_2)

		  hi2c.Instance = I2C2;
		  hi2c.Init.ClockSpeed = 100000;
		  hi2c.Init.DutyCycle = I2C_DUTYCYCLE_2;
		  hi2c.Init.OwnAddress1 = 0;
		  hi2c.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
		  hi2c.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
		  hi2c.Init.OwnAddress2 = 0;
		  hi2c.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
		  hi2c.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
		  if (HAL_I2C_Init(&hi2c) != HAL_OK)
		  {
			  ret = EEPROM_INIT_ERROR ;
		  }
#endif // I2C_2

		  /*  EEPROM  is connected to the Bus or NOT */
		  hal_stat = HAL_I2C_IsDeviceReady(&hi2c, EEPROM_READ_ADDRESS(eeprom_address), 2 , TIME_OUT);

		  if (hal_stat != HAL_OK)
		  {
			  ret = EEPROM_ADDRESS_NOT_FOUND ;
		  }else{
			  ret = EEPROM_OK ;
		      } // else status
	}// else
return ret ;
}
/*
 *  @brief : read some block in memory with its base address and chose an offset
 *  @param : eeprom_instace eeprom struct that contain the address of eeprom
 *
 */
std_return_type eeprom_read(const eeprom_t *eeprom_instace , uint16_t address ,
							uint16_t offset,  uint32_t *read_value)
{
	std_return_type ret = EEPROM_OK;
	if (NULL == eeprom_instace || NULL == read_value )
	{
		ret = EEPROM_NULL_POINTER ;
	}else if (address > MAX_ADDRESS )
		{
			ret = EEPROM_ADDRESS_OUT_OF_BOUND ;
		}
	else {

		/* ----- Calculating EERPOM Address ----- */

		uint8_t eeprom_address  = EEPROM_ADDRESS(eeprom_instace->address.address_pins_total);

		/* ----- Calculating EERPOM Page Number Address ----- */
		//uint8_t page_number =  (address / EEPROM_PAGE_SIZE) ;
		//uint8_t address_loc =  (address % EEPROM_PAGE_SIZE) ;

		/*---- Create Data Buffer Variable  ----- */
		uint8_t data [offset]   ;

		/*---- READ data in EEprom and save it in data buffer Variable  ----- */

		uint8_t hal_stat = HAL_I2C_Mem_Read(&hi2c, EEPROM_READ_ADDRESS(eeprom_address), address,
											MEM_ADD_SIZE, data,
											offset,TIME_OUT) ;
		if (hal_stat != HAL_OK)
		{
			ret = EEPROM_READ_ERROR ;
		}else {
			/*---- Convert  Data Buffer Variable to Uint32 variable   ----- */
			byte_to_uint32(data , read_value) ;
			ret = EEPROM_OK ;
		}


	}
return ret ;
}
std_return_type eeprom_write(const eeprom_t *eeprom_instace , uint16_t address  , uint16_t offset , uint32_t write_value ){
	std_return_type ret = EEPROM_OK;
	if (NULL == eeprom_instace )
	{
		ret = EEPROM_NULL_POINTER ;
	}else if (address > MAX_ADDRESS )
	{
		ret = EEPROM_ADDRESS_OUT_OF_BOUND ;
	}
else {

	/* ----- Calculating EERPOM Address ----- */

	uint8_t eeprom_address  = EEPROM_ADDRESS(eeprom_instace->address.address_pins_total);

	/* ----- Calculating EERPOM Page Number Address ----- */
	//uint8_t page_number =  (address / EEPROM_PAGE_SIZE) ;
	//uint8_t address_loc =  (address % EEPROM_PAGE_SIZE) ;

	/*---- Create Data Buffer Variable  ----- */
	uint8_t data [offset]   ;

	/*---- Convert  Data Buffer Variable to Uint32 variable   ----- */
	uint32_to_byte(data, &write_value) ;
	/*---- READ data in EEprom and save it in data buffer Variable  ----- */

	uint8_t hal_stat = HAL_I2C_Mem_Write(&hi2c, EEPROM_READ_ADDRESS(eeprom_address),
										address,MEM_ADD_SIZE, data,
										offset,TIME_OUT) ;
	if (hal_stat != HAL_OK)
	{
		ret = EEPROM_WRITE_ERROR ;
	}else {


		ret = EEPROM_OK ;
	}


}
return ret ;
}

std_return_type eeprom_reset(const eeprom_t *eeprom_instace){
	std_return_type ret = EEPROM_OK;
		if (NULL == eeprom_instace)
		{
			ret = EEPROM_NULL_POINTER ;
		}else {


		}
	return ret ;
}

/*	---- Change array of Four Byte to uin23_t variable 	*/
static void byte_to_uint32 (uint8_t *ptr , uint32_t *num )
{
	/*---- Make Union Object ----*/
	union {
		uint8_t data [4] ;
		uint32_t value  ;
	}number ;

	/*----- Copy the Value of array to number Union  Object ----- */

	for (uint8_t  i = 0 ; i <4 ;  i++  )
	{
		number.data[i] = *(ptr+i) ;
	}
	*num = number.value ;
}

 static void uint32_to_byte (uint8_t *ptr , uint32_t *num )
{
	/*---- Make Union Object ----*/
	union {
		uint8_t data [4] ;
		uint32_t value  ;
	}number ;

	 number.value  = *num ;

	/*----- Copy the Value of number  to  array  ----- */

	for (uint8_t  i = 0 ; i <4 ;  i++  )
	{
		*(ptr+i) = number.data[i] ;
	}

}
