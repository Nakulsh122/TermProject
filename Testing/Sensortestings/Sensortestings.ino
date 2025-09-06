// Arduino (ESP32) sketch
// - Requires MPU6050 library (Jeff Rowberg style)
// - Emits CSV: timestamp_ms, ax_m/s2, ay_m/s2, az_m/s2, gx_dps, gy_dps, gz_dps
// - Calibrates biases at startup (keep sensor perfectly still)

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// calibration (raw counts)
float ax_bias = 0;
float ay_bias = 0;
float az_bias = 0;
float gx_bias = 0;
float gy_bias = 0;
float gz_bias = 0;

// timing
const unsigned long SAMPLE_US = 10000UL; // 10 ms -> 100 Hz

void calibrateSensor(int samples = 500) {
  long ax_sum = 0, ay_sum = 0, az_sum = 0;
  long gx_sum = 0, gy_sum = 0, gz_sum = 0;

  Serial.println("Starting calibration (keep sensor still)...");
  delay(200);

  for (int i = 0; i < samples; ++i) {
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

    ax_sum += ax_raw; ay_sum += ay_raw; az_sum += az_raw;
    gx_sum += gx_raw; gy_sum += gy_raw; gz_sum += gz_raw;
    delay(5);
  }

  ax_bias = (float)ax_sum / samples;
  ay_bias = (float)ay_sum / samples;
  // For Z, subtract 1g (in raw counts at ±2g it's ~16384)
  az_bias = ((float)az_sum / samples) - 16384.0;

  gx_bias = (float)gx_sum / samples;
  gy_bias = (float)gy_sum / samples;
  gz_bias = (float)gz_sum / samples;

  Serial.println("Calibration done.");
  Serial.print("ax_bias: "); Serial.println(ax_bias, 2);
  Serial.print("ay_bias: "); Serial.println(ay_bias, 2);
  Serial.print("az_bias(raw-1g): "); Serial.println(az_bias, 2);
  Serial.print("gx_bias: "); Serial.println(gx_bias, 2);
  Serial.print("gy_bias: "); Serial.println(gy_bias, 2);
  Serial.print("gz_bias: "); Serial.println(gz_bias, 2);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1) delay(1000);
  }

  // force ranges
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2); // ±2g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250); // ±250 dps
  // set DLPF low-pass for smoother accel (try mode 3 ~44Hz)
  mpu.setDLPFMode(3);
  // sample rate divider (gyro rate/(1+rate)), 4 -> ~200Hz base / (1+4) if needed
  mpu.setRate(4);

  delay(100);

  calibrateSensor(500);

  // print header (optional)
  // Serial.println("ts_ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  static unsigned long next_us = micros();
  unsigned long now_us = micros();
  if ((long)(now_us - next_us) < 0) {
    // not time yet
    return;
  }
  next_us += SAMPLE_US;

  int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
  mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

  // Apply raw biases (raw counts)
  float axc = (float)ax_raw - ax_bias;
  float ayc = (float)ay_raw - ay_bias;
  float azc = (float)az_raw - az_bias;

  float gxc = (float)gx_raw - gx_bias;
  float gyc = (float)gy_raw - gy_bias;
  float gzc = (float)gz_raw - gz_bias;

  // scale to SI
  const float ACCEL_SCALE = 16384.0; // LSB/g for ±2g
  const float G = 9.81;
  float ax_ms2 = (axc / ACCEL_SCALE) * G;
  float ay_ms2 = (ayc / ACCEL_SCALE) * G;
  float az_ms2 = (azc / ACCEL_SCALE) * G;

  const float GYRO_SCALE = 131.0; // LSB/(deg/s) for ±250 dps
  float gx_dps = gxc / GYRO_SCALE;
  float gy_dps = gyc / GYRO_SCALE;
  float gz_dps = gzc / GYRO_SCALE;

  unsigned long ts = millis();

  // Print CSV: timestamp(ms), ax, ay, az (m/s^2), gx, gy, gz (deg/s)
  Serial.print(ts); Serial.print(",");
  Serial.print(ax_ms2, 4); Serial.print(",");
  Serial.print(ay_ms2, 4); Serial.print(",");
  Serial.print(az_ms2, 4); Serial.print(",");
  Serial.print(gx_dps, 4); Serial.print(",");
  Serial.print(gy_dps, 4); Serial.print(",");
  Serial.println(gz_dps, 4);

  // Wait until next_us to keep stable rate
  // small sleep to yield CPU (not necessary but friendly)
  delayMicroseconds(100);
}
