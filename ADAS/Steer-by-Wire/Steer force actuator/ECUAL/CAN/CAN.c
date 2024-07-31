/**
 * @file CAN.c
 * @author Muhammad Osama elaraby (eng.muhammad.osama.9@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-18
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "CAN.h"

CAN_HandleTypeDef hcan;
CAN_FilterTypeDef TX_sFilterConfig;
CAN_FilterTypeDef RX_sFilterConfig;

uint32_t pTxMailbox;
uint8_t no_free_tx_box;
struct FRAME_LIST FirstFRAMETX;
struct FRAME_LIST FirstFRAMERX;
struct FRAME_LIST *gpFRAMETX = &FirstFRAMETX;
struct FRAME_LIST *gpFRAMERX = &FirstFRAMERX;
uint8_t TX_filter_num = 8;
uint8_t RX_filter_num;

struct FRAME_LIST frame_test;

uint32_t filter[14][7] = {
	{1, 0x666, 0x7ff, 0x000, 0x000, CAN_FILTERMODE_IDLIST, CAN_FILTERSCALE_16BIT}}; // 0-fifo(0 or 1)	1-ID1	2-MASk1(ID3)		3-ID2	4-MASk2(ID4)	5-FilterMode	6-FilterScale

void CAN_TX_REQUEST();
static void CAN_RX_INTERRUPT(struct FRAME_LIST *frame, uint32_t RxFifo);

// IRQ CALLBACKS
void HAL_CAN_TxMailbox0CompleteCallback(CAN_HandleTypeDef *hcan)
{
}
void HAL_CAN_TxMailbox1CompleteCallback(CAN_HandleTypeDef *hcan)
{
}
void HAL_CAN_TxMailbox2CompleteCallback(CAN_HandleTypeDef *hcan)
{
}
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
	struct FRAME_LIST *p_FRAME = FirstFRAMERX.PNextFRAME;
	struct FRAME_LIST temp;

	CAN_RX_INTERRUPT(&temp, CAN_RX_FIFO0);

	while (p_FRAME)
	{
		if (temp.FRAME.IDNUM == p_FRAME->FRAME.IDNUM)
		{
			p_FRAME->FRAME.DLC = temp.FRAME.DLC;
			p_FRAME->FRAME.data = temp.FRAME.data;
			break;
		}
		p_FRAME = p_FRAME->PNextFRAME;
	}
}
/**
 * @brief
 *
 * @param hcan
 */
void HAL_CAN_RxFifo1MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
	CAN_TxHeaderTypeDef pHeader;
	pHeader.RTR = (CAN_RI0R_RTR & hcan->Instance->sFIFOMailBox[CAN_RX_FIFO1].RIR);
	if (pHeader.RTR)
	{
		pHeader.StdId = (CAN_RI0R_STID & hcan->Instance->sFIFOMailBox[CAN_RX_FIFO1].RIR) >> CAN_TI0R_STID_Pos;

		struct FRAME_LIST *p_FRAME = FirstFRAMETX.PNextFRAME;
		while (p_FRAME)
		{
			if (pHeader.StdId == p_FRAME->FRAME.IDNUM)
			{
				CAN_TX(p_FRAME);
				break;
			}
			p_FRAME = p_FRAME->PNextFRAME;
		}
	}
	else
	{
		pHeader.StdId = (CAN_RI0R_STID & hcan->Instance->sFIFOMailBox[CAN_RX_FIFO1].RIR) >> CAN_TI0R_STID_Pos;

		struct FRAME_LIST *p_FRAME = FirstFRAMERX.PNextFRAME;
		while (p_FRAME)
		{
			if (pHeader.StdId == p_FRAME->FRAME.IDNUM)
			{
				CAN_RX_INTERRUPT(p_FRAME, CAN_RX_FIFO1);
				break;
			}
			p_FRAME = p_FRAME->PNextFRAME;
		}
	}
}
/**
 * @brief
 *
 * @param hcan
 */
void HAL_CAN_ErrorCallback(CAN_HandleTypeDef *hcan)
{
	if ((hcan->ErrorCode & HAL_CAN_ERROR_TX_ALST0) == HAL_CAN_ERROR_TX_ALST0) //!< TxMailbox 0 transmit failure due to arbitration lost
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_TX_TERR0) == HAL_CAN_ERROR_TX_TERR0) //!< TxMailbox 0 transmit failure due to transmit error
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_TX_ALST1) == HAL_CAN_ERROR_TX_ALST1) //!< TxMailbox 1 transmit failure due to arbitration lost
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_TX_TERR1) == HAL_CAN_ERROR_TX_TERR1) //!< TxMailbox 1 transmit failure due to transmit error
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_TX_ALST2) == HAL_CAN_ERROR_TX_ALST2) //!< TxMailbox 2 transmit failure due to arbitration lost
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_TX_TERR2) == HAL_CAN_ERROR_TX_TERR2) //!< TxMailbox 2 transmit failure due to transmit error
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_RX_FOV0) == HAL_CAN_ERROR_RX_FOV0) //!< Rx FIFO0 overrun error
	{
	}
	if ((hcan->ErrorCode & HAL_CAN_ERROR_RX_FOV1) == HAL_CAN_ERROR_RX_FOV1) //!< Rx FIFO1 overrun error
	{
	}
}

void USB_HP_CAN1_TX_IRQHandler(void)
{
	HAL_CAN_IRQHandler(&hcan);
}

/**
 * @brief This function handles USB low priority or CAN RX0 interrupts.
 */
void USB_LP_CAN1_RX0_IRQHandler(void)
{
	HAL_CAN_IRQHandler(&hcan);
}

/**
 * @brief This function handles CAN RX1 interrupt.
 */
void CAN1_RX1_IRQHandler(void)
{
	HAL_CAN_IRQHandler(&hcan);
}

/**
 * @brief This function handles CAN SCE interrupt.
 */
void CAN1_SCE_IRQHandler(void)
{
	HAL_CAN_IRQHandler(&hcan);
}

void CAN_Add_Filter(struct FRAME_LIST *FRAME, filter_type_t type)
{

	CAN_FilterTypeDef *PsFilterConfig;
	uint8_t dual = 0;
	struct FRAME_LIST *p_FRAME = FirstFRAMETX.PNextFRAME;
	uint8_t *filter_num = 0;
	if (type == transmit)
	{

		gpFRAMETX->PNextFRAME = FRAME;
		FRAME->PNextFRAME = NULL;
		gpFRAMETX = FRAME;
		filter_num = &TX_filter_num;
		PsFilterConfig = &TX_sFilterConfig;
		if ((*filter_num) % 4 == 0)
		{
			PsFilterConfig->FilterIdLow = 0;
			PsFilterConfig->FilterIdHigh = 0;
			PsFilterConfig->FilterMaskIdHigh = 0;
			PsFilterConfig->FilterMaskIdLow = 0;
		}

		PsFilterConfig->FilterFIFOAssignment = CAN_FILTER_FIFO1;
	}
	else if ((type == receive))
	{
		while (p_FRAME)
		{
			if (FRAME->FRAME.IDNUM == p_FRAME->FRAME.IDNUM)
			{
				dual = 1;
				break;
			}
			p_FRAME = p_FRAME->PNextFRAME;
		}
		gpFRAMERX->PNextFRAME = FRAME;
		FRAME->PNextFRAME = NULL;
		gpFRAMERX = FRAME;
		filter_num = &RX_filter_num;
		PsFilterConfig = &RX_sFilterConfig;
		if ((*filter_num) % 4 == 0)
		{
			PsFilterConfig->FilterIdLow = 0;
			PsFilterConfig->FilterIdHigh = 0;
			PsFilterConfig->FilterMaskIdHigh = 0;
			PsFilterConfig->FilterMaskIdLow = 0;
		}

		PsFilterConfig->FilterFIFOAssignment = CAN_FILTER_FIFO0;
	}

	//	 (#) Configure the reception filters using the following configuration
	//      functions:
	//        (++) HAL_CAN_ConfigFilter()

	PsFilterConfig->FilterActivation = CAN_FILTER_ENABLE;
	PsFilterConfig->FilterBank = *filter_num / 4;
	PsFilterConfig->FilterMode = CAN_FILTERMODE_IDLIST;
	PsFilterConfig->FilterScale = CAN_FILTERSCALE_16BIT;
	if ((*filter_num) % 4 == 0)
	{
		PsFilterConfig->FilterIdLow = FRAME->FRAME.IDNUM << 5;
	}
	else if ((*filter_num) % 4 == 1)
	{
		PsFilterConfig->FilterIdHigh = FRAME->FRAME.IDNUM << 5;
	}
	else if ((*filter_num) % 4 == 2)
	{
		PsFilterConfig->FilterMaskIdHigh = FRAME->FRAME.IDNUM << 5;
	}
	else if ((*filter_num) % 4 == 3)
	{
		PsFilterConfig->FilterMaskIdLow = FRAME->FRAME.IDNUM << 5;
	}

	if (dual == 0)
	{
		(*filter_num)++;
		HAL_CAN_ConfigFilter(&hcan, PsFilterConfig);
	}
}

void HAL_CAN_MspInit(CAN_HandleTypeDef *hcan)
{
	GPIO_InitTypeDef GPIO_InitStruct = {0};
	if (hcan->Instance == CAN1)
	{
		/* USER CODE BEGIN CAN1_MspInit 0 */

		/* USER CODE END CAN1_MspInit 0 */
		/* Peripheral clock enable */
		__HAL_RCC_CAN1_CLK_ENABLE();

		__HAL_RCC_GPIOA_CLK_ENABLE();
		/**CAN GPIO Configuration
	PA11     ------> CAN_RX
	PA12     ------> CAN_TX
		 */
		GPIO_InitStruct.Pin = GPIO_PIN_11;
		GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
		GPIO_InitStruct.Pull = GPIO_NOPULL;
		HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

		GPIO_InitStruct.Pin = GPIO_PIN_12;
		GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
		GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
		HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

		/* CAN1 interrupt Init */
#ifdef CAN_INTERRUPTS_ENABLE
		HAL_NVIC_SetPriority(USB_HP_CAN1_TX_IRQn, 0, 0);
		HAL_NVIC_EnableIRQ(USB_HP_CAN1_TX_IRQn);
		HAL_NVIC_SetPriority(USB_LP_CAN1_RX0_IRQn, 0, 0);
		HAL_NVIC_EnableIRQ(USB_LP_CAN1_RX0_IRQn);
		HAL_NVIC_SetPriority(CAN1_RX1_IRQn, 0, 0);
		HAL_NVIC_EnableIRQ(CAN1_RX1_IRQn);
		HAL_NVIC_SetPriority(CAN1_SCE_IRQn, 0, 0);
		HAL_NVIC_EnableIRQ(CAN1_SCE_IRQn);
#endif
		/* USER CODE BEGIN CAN1_MspInit 1 */

		/* USER CODE END CAN1_MspInit 1 */
	}
}

void CAN_START()
{
	hcan.Instance = CAN1;
	hcan.Init.Prescaler = 4;
	hcan.Init.Mode = CAN_MODE_NORMAL;
	hcan.Init.SyncJumpWidth = CAN_SJW_1TQ;
	hcan.Init.TimeSeg1 = CAN_BS1_15TQ;
	hcan.Init.TimeSeg2 = CAN_BS2_2TQ;
	hcan.Init.TimeTriggeredMode = DISABLE;
	hcan.Init.AutoBusOff = DISABLE;
	hcan.Init.AutoWakeUp = DISABLE;
	hcan.Init.AutoRetransmission = DISABLE;
	hcan.Init.ReceiveFifoLocked = DISABLE;
	hcan.Init.TransmitFifoPriority = DISABLE;
	if (HAL_CAN_Init(&hcan) != HAL_OK)
	{
		Error_Handler();
	}

#ifdef CAN_INTERRUPTS_ENABLE
	HAL_CAN_ActivateNotification(&hcan, CAN_IT_TX_MAILBOX_EMPTY | CAN_IT_RX_FIFO0_MSG_PENDING | CAN_IT_RX_FIFO1_MSG_PENDING);
#endif

	if (HAL_CAN_Start(&hcan) != HAL_OK)
	{
		Error_Handler();
	}
}

CAN_TX_Handler CAN_TX(struct FRAME_LIST *frame)
{
	CAN_TxHeaderTypeDef pHeader;

	pHeader.StdId = frame->FRAME.IDNUM;
	pHeader.DLC = frame->FRAME.DLC;
	pHeader.IDE = CAN_ID_STD;
	pHeader.RTR = CAN_RTR_DATA;

	//(++) HAL_CAN_GetTxMailboxesFreeLevel() to get the number of free Tx mailboxes.

	no_free_tx_box = HAL_CAN_GetTxMailboxesFreeLevel(&hcan);
	if (no_free_tx_box)
	{
		//(++) HAL_CAN_AddTxMessage() to request transmission of a new message.

		if (HAL_CAN_AddTxMessage(&hcan, &pHeader, (uint8_t *)&(frame->FRAME.data.total), &pTxMailbox) != HAL_OK)
		{
			Error_Handler();
		}
	}
	else
	{
		return CAN_NO_FREE_MAILBOX;
	}

	return CAN_OK;
}

void CAN_RX()
{

	CAN_RxHeaderTypeDef pHeader;
	HAL_StatusTypeDef status = HAL_ERROR;
	while (HAL_CAN_GetRxFifoFillLevel(&hcan, CAN_RX_FIFO0) != 0)
	{

		struct FRAME_LIST *p_FRAME = FirstFRAMERX.PNextFRAME;

		pHeader.StdId = (CAN_RI0R_STID & hcan.Instance->sFIFOMailBox[CAN_RX_FIFO0].RIR) >> CAN_TI0R_STID_Pos;

		while (p_FRAME)
		{
			if (pHeader.StdId == p_FRAME->FRAME.IDNUM)
			{
				if (HAL_CAN_GetRxMessage(&hcan, CAN_RX_FIFO0, &pHeader, (uint8_t *)&(p_FRAME->FRAME.data.total)) == HAL_OK)
				{
					p_FRAME->FRAME.DLC = pHeader.DLC;
					status = HAL_OK;
				}
				break;
			}

			p_FRAME = p_FRAME->PNextFRAME;
		}
	}

	while (HAL_CAN_GetRxFifoFillLevel(&hcan, CAN_RX_FIFO1) != 0)
	{
		struct FRAME_LIST *p_FRAME;
		pHeader.RTR = (CAN_RI0R_RTR & hcan.Instance->sFIFOMailBox[CAN_RX_FIFO1].RIR);

		if (pHeader.RTR)
		{
			pHeader.StdId = (CAN_RI0R_STID & hcan.Instance->sFIFOMailBox[CAN_RX_FIFO1].RIR) >> CAN_TI0R_STID_Pos;

			p_FRAME = FirstFRAMETX.PNextFRAME;
			while (p_FRAME)
			{
				if (pHeader.StdId == p_FRAME->FRAME.IDNUM)
				{
					CAN_TX(p_FRAME);
					break;
				}
				p_FRAME = p_FRAME->PNextFRAME;
			}
		}
		else
		{
			pHeader.StdId = (CAN_RI0R_STID & hcan.Instance->sFIFOMailBox[CAN_RX_FIFO1].RIR) >> CAN_TI0R_STID_Pos;

			p_FRAME = FirstFRAMERX.PNextFRAME;
			while (p_FRAME)
			{
				if (pHeader.StdId == p_FRAME->FRAME.IDNUM)
				{
					if (HAL_CAN_GetRxMessage(&hcan, CAN_RX_FIFO1, &pHeader, (uint8_t *)&(p_FRAME->FRAME.data.total)) == HAL_OK)
					{
						p_FRAME->FRAME.DLC = pHeader.DLC;
						status = HAL_OK;
					}
					break;
				}
				p_FRAME = p_FRAME->PNextFRAME;
			}
		}
	}
}

static void CAN_RX_INTERRUPT(struct FRAME_LIST *frame, uint32_t RxFifo)
{
	CAN_RxHeaderTypeDef pHeader;
	if (HAL_CAN_GetRxMessage(&hcan, RxFifo, &pHeader, (uint8_t *)&(frame->FRAME.data.total)) != HAL_OK)
	{
		Error_Handler();
	}
	frame->FRAME.IDNUM = pHeader.StdId;
	frame->FRAME.DLC = pHeader.DLC;
}

void CAN_REQUEST(struct FRAME_LIST *frame)
{

	CAN_TxHeaderTypeDef pHeader;

	pHeader.StdId = frame->FRAME.IDNUM;
	pHeader.DLC = frame->FRAME.DLC;
	pHeader.IDE = CAN_ID_STD;
	pHeader.RTR = CAN_RTR_REMOTE;
	uint8_t data[8] = {0};
	//(++) HAL_CAN_GetTxMailboxesFreeLevel() to get the number of free Tx mailboxes.

	no_free_tx_box = HAL_CAN_GetTxMailboxesFreeLevel(&hcan);
	if (no_free_tx_box)
	{
		//(++) HAL_CAN_AddTxMessage() to request transmission of a new message.

		if (HAL_CAN_AddTxMessage(&hcan, &pHeader, data, &pTxMailbox) != HAL_OK)
		{
			Error_Handler();
		}
	}
}
