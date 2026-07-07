#ifndef __MY_IIC_H_
#define __MY_IIC_H_

#include "stm32f10x.h" 

#define I2Cx_SCL_PIN        GPIO_Pin_7
#define I2Cx_SCL_GPIO_PORT  GPIOB

#define I2Cx_SDA_PIN        GPIO_Pin_6
#define I2Cx_SDA_GPIO_PORT  GPIOB

#define I2C_SCL_HIGH()  GPIO_SetBits(I2Cx_SCL_GPIO_PORT, I2Cx_SCL_PIN)
#define I2C_SCL_LOW()   GPIO_ResetBits(I2Cx_SCL_GPIO_PORT, I2Cx_SCL_PIN)

#define I2C_SDA_HIGH()  GPIO_SetBits(I2Cx_SDA_GPIO_PORT, I2Cx_SDA_PIN)
#define I2C_SDA_LOW()   GPIO_ResetBits(I2Cx_SDA_GPIO_PORT, I2Cx_SDA_PIN)

#define I2C_SDA_READ()  GPIO_ReadInputDataBit(I2Cx_SDA_GPIO_PORT, I2Cx_SDA_PIN)

void i2c_delay(void);                          // I2C 延时
void i2c_config(void);                         // 初始化 IIC GPIO
void i2c_start(void);                          // 发送开始信号
void i2c_stop(void);                           // 发送停止信号
unsigned char i2c_send_byte(uint8_t data);     // 发送一个字节
uint8_t i2c_receive_byte(unsigned char ack);   // 接收一个字节
uint8_t i2c_wait_ack(void);                    // 等待 ACK
void i2c_ack(void);                            // 发送 ACK
void i2c_no_ack(void);                         // 发送 NACK
uint8_t I2CReceiveAck(uint8_t timeout);        // 接收 ACK (带超时)
void I2CReadBytes(int nBytes, char *dataP);    // 读取多个字节
void i2c_Wait(int t);                          // 简单延时

#endif
