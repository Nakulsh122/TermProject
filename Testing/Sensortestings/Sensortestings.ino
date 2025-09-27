  // Arduino (ESP32) sketch for MPU6050 + Python pipeline
  // - Outputs CSV: timestamp_ms, ax_m/s2, ay_m/s2, az_m/s2, gx_dps, gy_dps, gz_dps
  // - Calibrates at startup (sensor must be still)

  #include <Wire.h>
  #include <MPU6050.h>

  MPU6050 mpu;

  // calibration (raw counts)
  float ax_bias = 0, ay_bias = 0, az_bias = 0;
  float gx_bias = 0, gy_bias = 0, gz_bias = 0;

  void calibrateSensor(int samples = 500) {
    long ax_sum = 0, ay_sum = 0, az_sum = 0;
    long gx_sum = 0, gy_sum = 0, gz_sum = 0;

    Serial.println("Starting calibration (keep sensor still)...");
    delay(500);

    for (int i = 0; i < samples; ++i) {
      int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
      mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

      ax_sum += ax_raw;
      ay_sum += ay_raw;
      az_sum += az_raw;
      gx_sum += gx_raw;
      gy_sum += gy_raw;
      gz_sum += gz_raw;

      delay(5);
    }

    ax_bias = (float)ax_sum / samples;
    ay_bias = (float)ay_sum / samples;
    // subtract 1g (~16384 counts at ±2g)
    az_bias = ((float)az_sum / samples) - 16384.0;

    gx_bias = (float)gx_sum / samples;
    gy_bias = (float)gy_sum / samples;
    gz_bias = (float)gz_sum / samples;

    Serial.println("Calibration done.");
  }

  void setup() {
    Serial.begin(115200);
    Wire.begin(18, 19);


    mpu.initialize();
    if (!mpu.testConnection()) {
      Serial.println("MPU6050 connection failed!");
      while (1) delay(1000);
    }

    // Force ranges
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);   // ±2g
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);   // ±250 dps

    // Use DLPF for smoother readings (mode 3 = ~44 Hz cutoff)
    mpu.setDLPFMode(3);

    // Set sample rate = GyroRate / (1 + divider)
    // GyroRate = 1 kHz when DLPF enabled
    // Divider = 9 → 1000 / (1+9) = 100 Hz
    mpu.setRate(9);

    delay(100);
    calibrateSensor(500);

    // No CSV header, Python expects pure numbers
  }

  void loop() {
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

    // Apply biases
    float axc = (float)ax_raw - ax_bias;
    float ayc = (float)ay_raw - ay_bias;
    float azc = (float)az_raw - az_bias;
    float gxc = (float)gx_raw - gx_bias;
    float gyc = (float)gy_raw - gy_bias;
    float gzc = (float)gz_raw - gz_bias;

    // Convert to SI
    const float ACCEL_SCALE = 16384.0; // LSB/g @ ±2g
    const float G = 9.81;
    float ax_ms2 = (axc / ACCEL_SCALE) * G;
    float ay_ms2 = (ayc / ACCEL_SCALE) * G;
    float az_ms2 = (azc / ACCEL_SCALE) * G;

    const float GYRO_SCALE = 131.0; // LSB/(deg/s) @ ±250 dps
    float gx_dps = gxc / GYRO_SCALE;
    float gy_dps = gyc / GYRO_SCALE;
    float gz_dps = gzc / GYRO_SCALE;

    unsigned long ts = millis();

    // Print CSV line
    Serial.print(ts); Serial.print(",");
    Serial.print(ax_ms2, 4); Serial.print(",");
    Serial.print(ay_ms2, 4); Serial.print(",");
    Serial.print(az_ms2, 4); Serial.print(",");
    Serial.print(gx_dps, 4); Serial.print(",");
    Serial.print(gy_dps, 4); Serial.print(",");
    Serial.println(gz_dps, 4);

    // Wait 10 ms → 100 Hz
    delay(10);
  }
