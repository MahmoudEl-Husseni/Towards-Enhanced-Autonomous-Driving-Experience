/*
 * eeprom.h
 *
 *  Created on: Feb 19, 2024
 *      Author: engmu
 */

#ifndef EEPROM_EEPROM_H_
#define EEPROM_EEPROM_H_

/* ----Includes :sections ---- */
#include "../Util/util.h"
#include "eeprom_config.h"

/* ---- Macros :sections  ---- */



#if    defined(_24C01)
#define EEPROM_NUM_PAGE 16 	 // Number of Pages in EEPROM
#define EEPROM_PAGE_SIZE 8	 // Number of Byte in One Page
#define EEPROM_ADDRESS_SIZE 7 // address size in Bits


#elif  defined(_24C02)
#define EEPROM_NUM_PAGE 32 	 // Number of Pages in EEPROM
#define EEPROM_PAGE_SIZE 8	 // Number of Byte in One Page
#define EEPROM_ADDRESS_SIZE 8 // address size in Bits

#elif  defined(_24C04)

#define EEPROM_NUM_PAGE 32 	 // Number of Pages in EEPROM
#define EEPROM_PAGE_SIZE 16	 // Number of Byte in One Page
#define EEPROM_ADDRESS_SIZE 9 // address size in Bits

#elif  defined(_24C08)
#define EEPROM_NUM_PAGE 64 	 // Number of Pages in EEPROM
#define EEPROM_PAGE_SIZE 16	 // Number of Byte in One Page
#define EEPROM_ADDRESS_SIZE 10 // address size in Bits

#elif  defined(_24C016)
#define EEPROM_NUM_PAGE 128 	 // Number of Pages in EEPROM
#define EEPROM_PAGE_SIZE 16	 // Number of Byte in One Page
#define EEPROM_ADDRESS_SIZE 11 // address size in Bits
#endif
/*----------------------------------*/

#if  !defined (_24C01) && !defined (_24C02) && !defined (_24C04) && !defined (_24C08) && !defined (_24C016)
#warning "Plz Select EEPROM Type in <eeprom.h>"
#endif


#define FIRST_BYTE 0
#define SECOND_BYTE 8
#define THIRD_BYTE  16
#define FOURTH_BYTE 24

#define TIME_OUT 100

#define EEPROM_MSB_BASE_ADRESS 0x50 // 1010 <- MSB address line

#define EEPROM_ADDRESS(ADDRESS) (EEPROM_MSB_BASE_ADRESS | ADDRESS)

#define EEPROM_READ_ADDRESS(EEPROM_PIN_ADDRESS)  ((EEPROM_PIN_ADDRESS <<1) | 1 )

#define EEPROM_WRITE_ADDRESS(EEPROM_PIN_ADDRESS)  ((EEPROM_PIN_ADDRESS <<1)| 0 )

#define MAX_ADDRESS (EEPROM_NUM_PAGE * EEPROM_PAGE_SIZE)

#define MEM_ADD_SIZE 2

#define EEPROM_ERASE_DEFAULT_VAL 0xFF

/* ----  User define data types : Section     ---- */


typedef enum {
EEPROM_OK,
EEPROM_NULL_POINTER,
EEPROM_INIT_ERROR,
EEPROM_ADDRESS_NOT_FOUND,
EEPROM_ADDRESS_OUT_OF_BOUND,
EEPROM_READ_ERROR,
EEPROM_WRITE_ERROR,
EEPROM_I2C_BUS_SELECTION_ERROR,
EEPROM_PAGE_OUT_OF_BOUND ,
EEPROM_PAGE_ERASE_ERROR
}EEPROM_status_t;

typedef union  {
	struct {
		uint8_t A0 :1 ;  // <- write Logic on pin A0
		uint8_t A1 :1 ;  // <- write Logic on pin A1
		uint8_t A2 :1 ;  // <- write Logic on pin A2
		uint8_t RESERVED :5 ; // <- No Value Put here
	};
	uint8_t address_pins_total  ;
}eeprom_address_pins_t;


typedef  struct {
	eeprom_address_pins_t address ;
	uint8_t I2C_bus  ;

}eeprom_t;

/* ---- Function Declaration : Section 	*/
std_return_type eeprom_init(const eeprom_t *eeprom_instance);
std_return_type eeprom_reset(const eeprom_t *eeprom_instance);
std_return_type eeprom_read(const eeprom_t *eeprom_instance , uint16_t address ,
							uint16_t offset,
							int32_t *read_value) ;
std_return_type eeprom_write(const eeprom_t *eeprom_instance  ,uint16_t address
							, uint16_t offset ,
							int32_t write_value );

std_return_type eeprom_page_erase (const eeprom_t *eeprom_instance ,
									uint8_t page_number);

#endif /* EEPROM_EEPROM_H_ */
