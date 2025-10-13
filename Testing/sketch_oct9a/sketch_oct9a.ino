#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define ADXL345_ADDR 0x53

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);

  // Start measurement
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x2D); // POWER_CTL register
  Wire.write(0x08); // Measurement mode
  Wire.endTransmission();
}

void loop() {
  int16_t x, y, z;

  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x32); // DATAX0 register
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 6);

  x = Wire.read() | (Wire.read() << 8);
  y = Wire.read() | (Wire.read() << 8);
  z = Wire.read() | (Wire.read() << 8);

  Serial.print("X: "); Serial.print(x);
  Serial.print(" Y: "); Serial.print(y);
  Serial.print(" Z: "); Serial.println(z);

  delay(20);
}
