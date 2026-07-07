#include "my_iic.h"

#define frequent 2

// 简单延时
void i2c_Wait(int t)
{
    volatile int cnt; 
    while(t--)
    for(cnt=7; cnt>0; cnt--);
}

// 初始化 IIC
void i2c_config(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);  // 打开 GPIOB 时钟

    // 配置 SCL (PB7) 开漏输出
    GPIO_InitStructure.GPIO_Pin = I2Cx_SCL_PIN;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
    GPIO_Init(I2Cx_SCL_GPIO_PORT, &GPIO_InitStructure);

    // 配置 SDA (PB6) 开漏输出
    GPIO_InitStructure.GPIO_Pin = I2Cx_SDA_PIN;
    GPIO_Init(I2Cx_SDA_GPIO_PORT, &GPIO_InitStructure);

    // 拉高初始状态
    I2C_SCL_HIGH();
    I2C_SDA_HIGH();
}

void i2c_delay(void)
{
    i2c_Wait(frequent);
}

// I2C 起始信号
void i2c_start(void)
{
    I2C_SDA_HIGH();
    I2C_SCL_HIGH();
    i2c_Wait(frequent);

    I2C_SDA_LOW();
    i2c_Wait(frequent);

    I2C_SCL_LOW();
}

// I2C 停止信号
void i2c_stop(void)
{
    I2C_SCL_LOW();
    I2C_SDA_LOW();
    i2c_Wait(frequent);

    I2C_SCL_HIGH();
    i2c_Wait(frequent);

    I2C_SDA_HIGH();
    i2c_Wait(frequent);
}

// 发送 ACK
void i2c_ack(void)
{
    I2C_SCL_LOW();
    I2C_SDA_LOW();
    i2c_Wait(frequent);

    I2C_SCL_HIGH();
    i2c_Wait(frequent);

    I2C_SCL_LOW();
}

// 发送 NACK
void i2c_no_ack(void)
{
    I2C_SCL_LOW();
    I2C_SDA_HIGH();
    i2c_Wait(frequent);

    I2C_SCL_HIGH();
    i2c_Wait(frequent);

    I2C_SCL_LOW();
}

// 等待从机 ACK
uint8_t I2CReceiveAck(uint8_t timeout)
{
    I2C_SCL_LOW();
    I2C_SDA_HIGH();  

    i2c_Wait(frequent);

    while(timeout--)
    {
        if (I2C_SDA_READ() == Bit_RESET)
        {
            I2C_SCL_HIGH(); 
            i2c_Wait(frequent); 
            I2C_SCL_LOW();
            return 0; // 收到 ACK
        }
        i2c_Wait(frequent);
    }  

    I2C_SCL_HIGH(); 
    i2c_Wait(frequent);      
    I2C_SCL_LOW();
    i2c_Wait(frequent);  

    return 1; // 超时未收到 ACK
}

// 发送一个字节
unsigned char i2c_send_byte(uint8_t data)
{
    uint8_t i;
    for(i=0;i<8;i++)
    {
        I2C_SCL_LOW();
        if (data & 0x80)  I2C_SDA_HIGH();
        else              I2C_SDA_LOW();

        i2c_Wait(frequent);
        data <<= 1;

        I2C_SCL_HIGH();
        i2c_Wait(frequent);
    }

    I2C_SCL_LOW();
    I2C_SDA_HIGH();
    return I2CReceiveAck(200); // 等待 ACK
}

// 接收一个字节
uint8_t i2c_receive_byte(unsigned char ack)
{
    uint8_t i, byte = 0;
    I2C_SDA_HIGH(); // 释放 SDA

    for(i=0;i<8;i++)
    {
        byte <<= 1;
        I2C_SCL_LOW();
        i2c_Wait(frequent);

        I2C_SCL_HIGH();
        i2c_Wait(frequent);

        if (I2C_SDA_READ() == Bit_SET)
            byte |= 0x01;
    }

    I2C_SCL_LOW();
    if (ack) i2c_ack();
    else     i2c_no_ack();

    return byte;
}

// 连续读取多个字节
void I2CReadBytes(int nBytes, char *dataP)
{
    int i;
    for(i=0;i<nBytes;i++)
    {
        if(i == (nBytes-1))
            dataP[i] = i2c_receive_byte(0); // 最后一个字节发 NACK
        else
            dataP[i] = i2c_receive_byte(1); // 中间字节发 ACK
    }
}
