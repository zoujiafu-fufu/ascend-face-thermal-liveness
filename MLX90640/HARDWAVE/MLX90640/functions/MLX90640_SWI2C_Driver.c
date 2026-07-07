#include "My_IIC.h"
#include "MLX90640_I2C_Driver.h"


void MLX90640_I2CInit(void)
{
    i2c_config();  // 配置GPIO
}

int MLX90640_I2CWrite(uint8_t slaveAddr,uint16_t writeAddress, uint16_t data)
{
    uint8_t ack;

    i2c_start();
    // 写从机地址（写模式）
    ack = i2c_send_byte((slaveAddr << 1) | 0);
    if(ack) { i2c_stop(); return -1; }

    // 写寄存器地址（高字节）
    ack = i2c_send_byte((writeAddress >> 8) & 0xFF);
    if(ack) { i2c_stop(); return -1; }

    // 写寄存器地址（低字节）
    ack = i2c_send_byte(writeAddress & 0xFF);
    if(ack) { i2c_stop(); return -1; }

    // 写数据高字节
    ack = i2c_send_byte((data >> 8) & 0xFF);
    if(ack) { i2c_stop(); return -1; }

    // 写数据低字节
    ack = i2c_send_byte(data & 0xFF);
    if(ack) { i2c_stop(); return -1; }

    i2c_stop();
    return 0;
}

int MLX90640_I2CRead(uint8_t slaveAddr,uint16_t startAddress, uint16_t nMemAddressRead, uint16_t *data)
{
    uint8_t ack;
    uint16_t i;

    i2c_start();
    // 写从机地址（写模式）
    ack = i2c_send_byte((slaveAddr << 1) | 0);
    if(ack) { i2c_stop(); return -1; }

    // 写起始寄存器地址
    ack = i2c_send_byte((startAddress >> 8) & 0xFF);
    if(ack) { i2c_stop(); return -1; }
    ack = i2c_send_byte(startAddress & 0xFF);
    if(ack) { i2c_stop(); return -1; }

    // 重启信号，准备读
    i2c_start();
    ack = i2c_send_byte((slaveAddr << 1) | 1);
    if(ack) { i2c_stop(); return -1; }

    // 读取数据（MLX90640 每个寄存器 16bit = 两个字节，高字节在前）
    for(i = 0; i < nMemAddressRead; i++)
    {
        uint8_t high = i2c_receive_byte(1);   // 先读高字节
        uint8_t low  = i2c_receive_byte((i == nMemAddressRead-1) ? 0 : 1); 

        data[i] = ((uint16_t)high << 8) | low;
    }

    i2c_stop();
    return 0;
}

void MLX90640_I2CFreqSet(int freq)
{
    (void)freq;
}
