/**
 * @file util.c
 * @author Muhammad Osama elaraby (eng.muhammad.osama.9@gmail.com)
 * @brief some helper functions that i used in my project
 * @version 0.1
 * @date 2024-07-18
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "util.h"

// std_return_type float_to_int (float input , in32_t )

/**
 * @brief
 *
 * @param input
 * @param output
 * @return std_return_type
 */
std_return_type __round__(float input, int32_t *output)
{
	input += 0.5;
	*output = (int32_t)input;
	return __ROUND_OK;
}
/**
 * @brief
 *
 * @param input 
 * @param output
 * @param resoultion
 * @return std_return_type
 */
std_return_type __round_range(int32_t input, int32_t *output,
							  uint16_t resoultion)
{
	std_return_type ret;
	if (NULL == output)
	{
		ret = __ROUND_RANGE_NOT_OK;
	}
	else
	{
		if (input % resoultion >= (resoultion / 2))
		{
			*output += resoultion;
		}
		else
		{
			*output = input;
		}
		ret = __ROUND_RANGE_OK;
	}

	return ret;
}

/**
 * @brief Checks if an input value is within a specified range.
 *
 * This function determines whether the given input value falls within the range
 * specified by _min and _max (inclusive). If the input value is within the range,
 * the function sets the value pointed to by output to TRUE (non-zero). Otherwise,
 * it sets the value pointed to by output to FALSE (zero).
 *
 * @param _min The minimum value of the range.
 * @param _max The maximum value of the range.
 * @param input The value to check.
 * @param output Pointer to a uint8_t where the result will be stored.
 *               TRUE (non-zero) if the input is within the range, FALSE (zero) otherwise.
 *
 * @return Always returns 0.
 */

std_return_type in_between(int16_t _min, int16_t _max,
						   int16_t input, uint8_t *output)
{

	if (_min - ENDS_ERROR <= input && input <= _max + ENDS_ERROR)
	{
		*output = TRUE;
	}
	else
	{
		*output = FALSE;
	}

	return 0;
}

/**
 * @brief : linear Conversion between 2 ranges
 * @param  input : input value u want to Convert
 * @param output : output value after Conversion
 * @param ranges : refrence to < @map_range_t > struct which  put the ranges in it
 * @retval : return status
 * @Note : if the output equal to -1 that  check the return status
 */
std_return_type map(float input, float *output,
					map_range_t *ranges)
{
	std_return_type ret;
	/*---- NULL Pointer Checking ----	*/
	if (NULL == ranges || NULL == output)
	{

		ret = MAP_NOT_OK;
	}
	/*---- input value occurs in desired range checking  ----	*/

	else
	{
		float _input = fmax(ranges->input_min_range, fmin(input, ranges->input_max_range));
		/* ----  calculating  slope ---- */

		float slope = (ranges->output_max_range - ranges->output_min_range) / (ranges->input_max_range - ranges->input_min_range);

		/* ----  calculating  Offset  ---- */
		float offset = ranges->output_min_range - slope * ranges->input_min_range;

		/*---- Make Linear transformation ----	*/
		*output = _input * slope + offset;

		ret = MAP_OK;
	}
	return ret;
}
