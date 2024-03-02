/*
 * eeprom.c
 *
 *  Created on: Feb 19, 2024
 *      Author: Muhammad Osama Elaraby 
 */

#include "eeprom.h"

/* ----- I2C Initialization Variable  ---- */
I2C_HandleTypeDef hi2c;

/* ----- Private Functions ----*/
static void byte_to_uint32 (uint8_t *ptr , uint32_t *num ) ;
static void uint32_to_byte (uint8_t *ptr , uint32_t *num );

std_return_type eeprom_init(const eeprom_t *eeprom_instance){
	std_return_type ret = EEPROM_OK;
	if (NULL == eeprom_instance)
	{
		ret = EEPROM_NULL_POINTER ;
	}else {
		uint8_t eeprom_address  = EEPROM_ADDRESS(eeprom_instance->address.address_pins_total);
		//uint8_t eeprom_address  = EEPROM_ADDRESS(0);

		/*HAL status Variable */
		uint8_t hal_stat ;

		/* ------ Select i2c Bus ---- */

		if (eeprom_instance->I2C_bus == I2C_BUS_1)
				{
					hi2c.Instance = I2C1;
				}
				else if (eeprom_instance->I2C_bus == I2C_BUS_2)
				{
					hi2c.Instance = I2C2;
				}
				else {
						ret = EEPROM_I2C_BUS_SELECTION_ERROR ;

				}
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


		  /*  EEPROM  is connected to the Bus or NOT */
		  hal_stat = HAL_I2C_IsDeviceReady(&hi2c, EEPROM_READ_ADDRESS(eeprom_address), 10 , TIME_OUT);
		 // hal_stat = HAL_I2C_IsDeviceReady(&hi2c, (eeprom_address), 10 , TIME_OUT);

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
 *  @param : eeprom_instance eeprom struct that contain the address of eeprom
 *
 */
std_return_type eeprom_read(const eeprom_t *eeprom_instance , uint16_t address ,
							uint16_t offset,  uint32_t *read_value)
{
	std_return_type ret = EEPROM_OK;
	if (NULL == eeprom_instance || NULL == read_value )
	{
		ret = EEPROM_NULL_POINTER ;
	}else if (address > MAX_ADDRESS )
		{
			ret = EEPROM_ADDRESS_OUT_OF_BOUND ;
		}
	else {

		/* ----- Calculating EERPOM Address ----- */

		uint8_t eeprom_address  = EEPROM_ADDRESS(eeprom_instance->address.address_pins_total);

		/* ----- Calculating EERPOM Page Number Address ----- */
		//uint8_t page_number =  (address / EEPROM_PAGE_SIZE) ;
		//uint8_t address_loc =  (address % EEPROM_PAGE_SIZE) ;

		/*---- Create Data Buffer Variable  ----- */
		uint8_t data [offset]   ;

		/*---- READ data in EEprom and save it in data buffer Variable  ----- */

		uint8_t hal_stat = HAL_I2C_Mem_Read(&hi2c, EEPROM_READ_ADDRESS(eeprom_address), address,
											MEM_ADD_SIZE, data,
											offset,TIME_OUT) ;
		HAL_Delay(2) ;
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

std_return_type eeprom_write(const eeprom_t *eeprom_instance , uint16_t address  , uint16_t offset , uint32_t write_value ){
	std_return_type ret = EEPROM_OK;
	if (NULL == eeprom_instance )
	{
		ret = EEPROM_NULL_POINTER ;
	}else if (address > MAX_ADDRESS )
	{
		ret = EEPROM_ADDRESS_OUT_OF_BOUND ;
	}
else {

	/* ----- Calculating EERPOM Address ----- */

	uint8_t eeprom_address  = EEPROM_ADDRESS(eeprom_instance->address.address_pins_total);

	/* ----- Calculating EERPOM Page Number Address ----- */
	//uint8_t page_number =  (address / EEPROM_PAGE_SIZE) ;
	//uint8_t address_loc =  (address % EEPROM_PAGE_SIZE) ;

	/*---- Create Data Buffer Variable  ----- */
	uint8_t data [offset]   ;

	/*---- Convert  Data Buffer Variable to Uint32 variable   ----- */
	uint32_to_byte(data, &write_value) ;
	/*---- READ data in EEprom and save it in data buffer Variable  ----- */

	uint8_t hal_stat = HAL_I2C_Mem_Write(&hi2c, EEPROM_WRITE_ADDRESS(eeprom_address),
										address,MEM_ADD_SIZE, data,
										offset,TIME_OUT) ;
	HAL_Delay(2) ;
	if (hal_stat != HAL_OK)
	{
		ret = EEPROM_WRITE_ERROR ;
	}else {


		ret = EEPROM_OK ;
	}


}
return ret ;
}



std_return_type eeprom_reset(const eeprom_t *eeprom_instance){
	std_return_type ret = EEPROM_OK;
		if (NULL == eeprom_instance)
		{
			ret = EEPROM_NULL_POINTER ;
		}else {

			for (uint8_t i=0 ; i<32 ; i ++ )
			{
					ret = eeprom_page_erase(eeprom_instance, i);
					if (ret!= EEPROM_OK )
					{
						ret =EEPROM_PAGE_ERASE_ERROR ;
						break ;
					}
			}

		}
	return ret ;
}

std_return_type eeprom_page_erase (const eeprom_t *eeprom_instance ,
									uint8_t page_number)
{
	std_return_type ret = EEPROM_OK;
			if (NULL == eeprom_instance)
			{
				ret = EEPROM_NULL_POINTER ;
			}else {
				if (page_number > EEPROM_NUM_PAGE)
				{
						ret = EEPROM_PAGE_OUT_OF_BOUND ;
				}
				else {
					uint16_t page_start_address = (page_number-1) * EEPROM_PAGE_SIZE ;

					ret = eeprom_write(eeprom_instance, page_start_address,EEPROM_PAGE_SIZE , EEPROM_ERASE_DEFAULT_VAL);

					HAL_Delay(2) ;

					if (ret != EEPROM_OK)
					{
						ret = EEPROM_PAGE_ERASE_ERROR ;
					}
				}

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

/*	---- Change  uin23_t variable to array of 4 bytes  	*/
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
