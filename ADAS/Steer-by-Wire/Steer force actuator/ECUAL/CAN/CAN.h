/*
 * CAN.h
 *
 *  Created on: Feb 3, 2024
 *      Author: Ziad Mahmoud Saad
 */

#ifndef INC_CAN_H_
#define INC_CAN_H_

#include "stm32f1xx_hal.h"


//#define CAN_INTERRUPTS_ENABLE

typedef union {
struct {
    uint8_t byte1  ;
    uint8_t byte2 ;
    uint8_t byte3   ;
    uint8_t byte4  ;
    uint8_t byte5  ;
    uint8_t byte6   ;
    uint8_t byte7 ;
    uint8_t byte8   ;
}  ;
uint64_t total  ;
}data_t;

typedef struct {
	uint32_t IDNUM;
	uint8_t DLC ;
	data_t data;
}ID_conf_t;

struct FRAME_LIST
{
	ID_conf_t FRAME;
	struct FRAME_LIST* PNextFRAME ;
};

typedef enum {
	transmit,
	receive
}filter_type_t;


typedef enum {
	CAN_OK,
	CAN_NOT_OK,
	CAN_NO_FREE_MAILBOX
}CAN_TX_Handler;


typedef uint8_t std_return_type ;
/* Section  : Functions Declaration  	*/

void CAN_START();
CAN_TX_Handler CAN_TX(struct FRAME_LIST* frame);
void CAN_RX();
void CAN_REQUEST(struct FRAME_LIST* frame);
void CAN_Add_Filter(struct FRAME_LIST *FRAME,filter_type_t type);
#endif /* INC_CAN_H_ */
