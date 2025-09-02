#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }

  // ±2g accel, ±250 dps gyro
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
}

void loop() {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Scale factors
  const float accelScale = 16384.0; // LSB/g
  const float gyroScale  = 131.0;   // LSB/(°/s)

  float ax_ms2 = (float)ax / accelScale * 9.81;
  float ay_ms2 = (float)ay / accelScale * 9.81;
  float az_ms2 = (float)az / accelScale * 9.81;

  float gx_dps = (float)gx / gyroScale;
  float gy_dps = (float)gy / gyroScale;
  float gz_dps = (float)gz / gyroScale;

  unsigned long ts = millis();

  // Print CSV: timestamp, ax, ay, az, gx, gy, gz
  Serial.print(ts); Serial.print(",");
  Serial.print(ax_ms2, 3); Serial.print(",");
  Serial.print(ay_ms2, 3); Serial.print(",");
  Serial.print(az_ms2, 3); Serial.print(",");
  Serial.print(gx_dps, 3); Serial.print(",");
  Serial.print(gy_dps, 3); Serial.print(",");
  Serial.println(gz_dps, 3);

  delay(10); // ~100 Hz
}
