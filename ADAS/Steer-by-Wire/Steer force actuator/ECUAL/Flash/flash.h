/*
 * flash.h
 *
 *  Created on: Apr 18, 2024
 *      Author: engmu
 */

#ifndef FLASH_FLASH_H_
#define FLASH_FLASH_H_

#include "../Util/util.h"

/*
 * last sector  address of page (1k byte) in flash
 *  FLASH_ST_ADD = 0x08000000
 *  last PAGE Address = 0x08000000 + 63k byte  = 0x800FC00
 *  */
#define PAGE_ADD  0x800FC04
typedef uint8_t std_falsh_return ;

typedef enum {
	FLASH_OK ,
	FLASH_NOT_OK
}flash_std;

std_falsh_return falsh_read  (uint32_t *data ) ;
std_falsh_return flash_write (uint32_t data);
std_falsh_return flash_erase () ;



#endif /* FLASH_FLASH_H_ */
