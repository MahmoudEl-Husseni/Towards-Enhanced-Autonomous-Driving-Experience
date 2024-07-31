/**
 * @file flash.c
 * @author Muhammad Osama elaraby (eng.muhammad.osama.9@gmail.com)
 * @brief driver for clear and write in flash memory
 * @version 0.1
 * @date 2024-07-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "flash.h"

std_return_type falsh_read(uint32_t *data)
{
	/* Read Value Stored in 0x800FC00 Address in Flash   */
	*data = (uint32_t) * ((volatile uint32_t *)PAGE_ADD);
	return FLASH_OK;
}
std_return_type flash_write(uint32_t data)
{
	std_return_type ret;
	HAL_StatusTypeDef flash_st;
	HAL_FLASH_Unlock();
	flash_st = HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, PAGE_ADD, data);
	HAL_FLASH_Lock();
	ret = flash_st;
	return ret;
}
std_return_type flash_erase()
{

	std_return_type ret = FLASH_NOT_OK;
	FLASH_EraseInitTypeDef flash_erase;
	uint32_t error;
	HAL_FLASH_Unlock();
	flash_erase.TypeErase = FLASH_TYPEERASE_PAGES;
	flash_erase.PageAddress = PAGE_ADD;
	flash_erase.Banks = FLASH_BANK_1;
	flash_erase.NbPages = 1;
	HAL_FLASHEx_Erase(&flash_erase, &error);
	HAL_FLASH_Lock();
	if (error)
	{
		ret = FLASH_OK;
	}
	return ret;
}
