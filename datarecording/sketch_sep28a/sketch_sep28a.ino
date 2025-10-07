/*
  ESP32 + MPU6050 streaming while button pressed
  - Uses Adafruit_MPU6050 library
  - Streams CSV: ms_since_start, ax, ay, az, gx, gy, gz
  - Button active LOW (INPUT_PULLUP). Press = stream.
  - Simple startup calibration (1 second) to compute biases.
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// --- User config ---
const int SDA_PIN = 21;               // ESP32 SDA
const int SCL_PIN = 22;               // ESP32 SCL
const int BUTTON_PIN = 14;              // Button pin (to GND). Active LOW
const int LED_PIN = 2;                // status LED (optional)
const float SAMPLE_RATE_HZ = 100.0;   // sampling rate
const unsigned long CALIB_MS = 1000;  // calibration duration
// --------------------

volatile bool streaming = false;

float accelBias[3] = { 0, 0, 0 };
float gyroBias[3] = { 0, 0, 0 };

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Initialize I2C explicitly (ESP32 requirement for custom pins)
  Wire.begin(SDA_PIN, SCL_PIN);

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found. Check wiring.");
    while (1) {
      digitalWrite(LED_PIN, millis() % 500 < 250);  // blink fast error
      delay(10);
    }
  }

  // Configure sensor settings - tune if desired
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);  // ±4g
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);       // ±500 °/s
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);    // low-pass to reduce noise

  Serial.println("MPU6050 found and configured.");
  Serial.print("Calibrating for ");
  Serial.print(CALIB_MS);
  Serial.println(" ms. Keep sensor still...");

  calibrateSensors();

  Serial.println("CAL_DONE");
  Serial.println("Format: ms, ax(g), ay(g), az(g), gx(deg/s), gy(deg/s), gz(deg/s)");
  delay(200);
}

void loop() {
  static unsigned long lastMicros = 0;
  const unsigned long samplePeriodUs = (unsigned long)(1e6 / SAMPLE_RATE_HZ);

  // Debounced button read (simple)
  bool btnPressed = (digitalRead(BUTTON_PIN) == LOW);  // active LOW
  streaming = btnPressed;

  digitalWrite(LED_PIN, streaming ? HIGH : LOW);

  unsigned long now = micros();
  if (now - lastMicros >= samplePeriodUs) {
    lastMicros += samplePeriodUs;  // keep phase (better than setting to now)
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // a.acceleration is m/s^2; convert to g for compactness (optional)
    float ax_g = a.acceleration.x / 9.80665f - accelBias[0];
    float ay_g = a.acceleration.y / 9.80665f - accelBias[1];
    float az_g = a.acceleration.z / 9.80665f - accelBias[2];

    // g.gyro is in rad/s in some Adafruit versions; convert carefully:
    // Adafruit_MPU6050 returns gyro in rad/s in sensors_event_t.gyro (per docs).
    // Convert to deg/s:
    float gx_dps = g.gyro.x * 57.295779513f - gyroBias[0];
    float gy_dps = g.gyro.y * 57.295779513f - gyroBias[1];
    float gz_dps = g.gyro.z * 57.295779513f - gyroBias[2];

    // Only stream when button pressed
    if (streaming) {
      unsigned long ms = millis();
      // CSV line:
      Serial.print(ms);
      Serial.print(',');
      Serial.print(ax_g, 5);
      Serial.print(',');
      Serial.print(ay_g, 5);
      Serial.print(',');
      Serial.print(az_g, 5);
      Serial.print(',');
      Serial.print(gx_dps, 4);
      Serial.print(',');
      Serial.print(gy_dps, 4);
      Serial.print(',');
      Serial.print(gz_dps, 4);
      Serial.println();
    }
  } else {
    // small yield to allow background tasks
    // do not delay large amounts as it will affect sampling timing
    yield();
  }
}

// -------------------------------------------------
// Simple bias calibration: average sensor while still
// -------------------------------------------------
void calibrateSensors() {
  unsigned long start = millis();
  unsigned long end = start + CALIB_MS;
  const int maxSamples = 2000;
  int cnt = 0;
  double ax_sum = 0, ay_sum = 0, az_sum = 0;
  double gx_sum = 0, gy_sum = 0, gz_sum = 0;

  while (millis() < end && cnt < maxSamples) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax_sum += a.acceleration.x / 9.80665f;
    ay_sum += a.acceleration.y / 9.80665f;
    az_sum += a.acceleration.z / 9.80665f;
    gx_sum += g.gyro.x * 57.295779513f;
    gy_sum += g.gyro.y * 57.295779513f;
    gz_sum += g.gyro.z * 57.295779513f;
    cnt++;
    delay(5);  // ~200 Hz sampling during calibration (fine)
  }
  if (cnt == 0) cnt = 1;
  accelBias[0] = (float)(ax_sum / cnt);
  accelBias[1] = (float)(ay_sum / cnt);
  accelBias[2] = (float)(az_sum / cnt) - 1.0f;  // remove gravity effect on Z (assuming Z up)
  gyroBias[0] = (float)(gx_sum / cnt);
  gyroBias[1] = (float)(gy_sum / cnt);
  gyroBias[2] = (float)(gz_sum / cnt);

  Serial.print("Calib samples: ");
  Serial.println(cnt);
  Serial.print("Accel bias (g): ");
  Serial.print(accelBias[0], 6);
  Serial.print(", ");
  Serial.print(accelBias[1], 6);
  Serial.print(", ");
  Serial.println(accelBias[2], 6);
  Serial.print("Gyro bias (deg/s): ");
  Serial.print(gyroBias[0], 4); Serial.print(",");
}