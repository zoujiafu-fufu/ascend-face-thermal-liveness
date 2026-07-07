#include "stm32f10x.h"
#include "delay.h"
#include "usart.h"
#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"
#include <stdio.h>

#define MLX90640_ADDR 0x33  
#define  FPS2HZ   0x02
#define  FPS4HZ   0x03
#define  FPS8HZ   0x04
#define  FPS16HZ  0x05
#define  FPS32HZ  0x06

paramsMLX90640 mlx90640;
float mlx90640To[768];     
uint16_t eeData[832];      
uint16_t frameData[834];   


int main(void)
{
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    delay_init();
    uart_init(115200);
	printf("USART1 OK\r\n");
    MLX90640_I2CInit();
    MLX90640_SetRefreshRate(MLX90640_ADDR, FPS8HZ);		
		MLX90640_SetChessMode(MLX90640_ADDR);	
		MLX90640_DumpEE(MLX90640_ADDR, eeData);						
		MLX90640_ExtractParameters(eeData, &mlx90640);		

    while (1)
    {
				for (unsigned char  page = 0 ;page < 2;page++)
				{
						int status = MLX90640_GetFrameData(MLX90640_ADDR, frameData);
						float Ta = MLX90640_GetTa(frameData, &mlx90640); 
						float tr = Ta - 8;
						MLX90640_CalculateTo_float(frameData, &mlx90640, 0.95, tr, mlx90640To);
				}
        for (int row = 0; row < 24; row++)
        {
            for (int col = 0; col < 32; col++)
            {
                printf("%6.2f ", mlx90640To[row * 32 + col]);
            }
            printf("\r\n");
        }
        printf("\r\n");

        delay_ms(100); 
    }
}
