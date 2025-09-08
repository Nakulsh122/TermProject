// Enhanced Arduino (ESP32) sketch for MPU6050 + Improved Python pipeline
// - Outputs CSV: timestamp_ms, ax_m/s2, ay_m/s2, az_m/s2, gx_dps, gy_dps, gz_dps, temperature_C, status
// - Multi-stage calibration with stability checking
// - Built-in data validation and filtering
// - Temperature monitoring for drift compensation
// - Status reporting for data quality assessment

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// Enhanced calibration parameters
struct CalibrationData {
  float ax_bias = 0, ay_bias = 0, az_bias = 0;
  float gx_bias = 0, gy_bias = 0, gz_bias = 0;
  float ax_variance = 0, ay_variance = 0, az_variance = 0;
  float gx_variance = 0, gy_variance = 0, gz_variance = 0;
  bool is_valid = false;
  float baseline_temp = 0;
} calibration;

// Data quality monitoring
struct DataQuality {
  unsigned long total_samples = 0;
  unsigned long invalid_samples = 0;
  float last_ax = 0, last_ay = 0, last_az = 0;
  float last_gx = 0, last_gy = 0, last_gz = 0;
  bool first_reading = true;
} quality;

// Configuration constants
const int CALIBRATION_SAMPLES = 1000;
const int STABILITY_CHECK_SAMPLES = 100;
const float MAX_ACCEL_VARIANCE = 0.1;     // m/s² squared
const float MAX_GYRO_VARIANCE = 2.0;      // (deg/s) squared
const float MAX_REASONABLE_ACCEL = 50.0;  // m/s²
const float MAX_REASONABLE_GYRO = 1000.0; // deg/s
const float TEMP_DRIFT_THRESHOLD = 5.0;   // °C
const float SPIKE_DETECTION_THRESHOLD = 20.0; // for acceleration spikes

// Moving average filter for temperature compensation
class MovingAverageFilter {
private:
  float* buffer;
  int size;
  int index;
  float sum;
  bool filled;

public:
  MovingAverageFilter(int window_size) {
    size = window_size;
    buffer = new float[size];
    index = 0;
    sum = 0;
    filled = false;
    for (int i = 0; i < size; i++) {
      buffer[i] = 0;
    }
  }
  
  ~MovingAverageFilter() {
    delete[] buffer;
  }
  
  float update(float value) {
    sum -= buffer[index];
    buffer[index] = value;
    sum += value;
    index = (index + 1) % size;
    if (!filled && index == 0) filled = true;
    
    return sum / (filled ? size : index + 1);
  }
  
  bool isReady() {
    return filled;
  }
};

// Temperature filter for stable readings
MovingAverageFilter tempFilter(10);

bool checkSensorStability(int samples = STABILITY_CHECK_SAMPLES) {
  float ax_sum = 0, ay_sum = 0, az_sum = 0;
  float gx_sum = 0, gy_sum = 0, gz_sum = 0;
  float ax_sq_sum = 0, ay_sq_sum = 0, az_sq_sum = 0;
  float gx_sq_sum = 0, gy_sq_sum = 0, gz_sq_sum = 0;
  
  Serial.println("Checking sensor stability...");
  
  for (int i = 0; i < samples; i++) {
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);
    
    // Convert to engineering units for stability check
    const float ACCEL_SCALE = 16384.0;
    const float G = 9.81;
    const float GYRO_SCALE = 131.0;
    
    float ax = ((float)ax_raw / ACCEL_SCALE) * G;
    float ay = ((float)ay_raw / ACCEL_SCALE) * G;
    float az = ((float)az_raw / ACCEL_SCALE) * G;
    float gx = (float)gx_raw / GYRO_SCALE;
    float gy = (float)gy_raw / GYRO_SCALE;
    float gz = (float)gz_raw / GYRO_SCALE;
    
    // Accumulate for variance calculation
    ax_sum += ax; ay_sum += ay; az_sum += az;
    gx_sum += gx; gy_sum += gy; gz_sum += gz;
    ax_sq_sum += ax*ax; ay_sq_sum += ay*ay; az_sq_sum += az*az;
    gx_sq_sum += gx*gx; gy_sq_sum += gy*gy; gz_sq_sum += gz*gz;
    
    delay(5);
  }
  
  // Calculate variances
  float ax_var = (ax_sq_sum/samples) - pow(ax_sum/samples, 2);
  float ay_var = (ay_sq_sum/samples) - pow(ay_sum/samples, 2);
  float az_var = (az_sq_sum/samples) - pow(az_sum/samples, 2);
  float gx_var = (gx_sq_sum/samples) - pow(gx_sum/samples, 2);
  float gy_var = (gy_sq_sum/samples) - pow(gy_sum/samples, 2);
  float gz_var = (gz_sq_sum/samples) - pow(gz_sum/samples, 2);
  
  // Check if sensor is stable enough
  bool stable = (ax_var < MAX_ACCEL_VARIANCE && ay_var < MAX_ACCEL_VARIANCE && az_var < MAX_ACCEL_VARIANCE &&
                 gx_var < MAX_GYRO_VARIANCE && gy_var < MAX_GYRO_VARIANCE && gz_var < MAX_GYRO_VARIANCE);
  
  if (stable) {
    Serial.println("Sensor is stable. Proceeding with calibration.");
  } else {
    Serial.println("Warning: Sensor readings are unstable!");
    Serial.print("Accel variances: "); Serial.print(ax_var, 6); Serial.print(", "); 
    Serial.print(ay_var, 6); Serial.print(", "); Serial.println(az_var, 6);
    Serial.print("Gyro variances: "); Serial.print(gx_var, 6); Serial.print(", "); 
    Serial.print(gy_var, 6); Serial.print(", "); Serial.println(gz_var, 6);
  }
  
  return stable;
}

void performEnhancedCalibration(int samples = CALIBRATION_SAMPLES) {
  long ax_sum = 0, ay_sum = 0, az_sum = 0;
  long gx_sum = 0, gy_sum = 0, gz_sum = 0;
  long ax_sq_sum = 0, ay_sq_sum = 0, az_sq_sum = 0;
  long gx_sq_sum = 0, gy_sq_sum = 0, gz_sq_sum = 0;
  
  Serial.print("Starting enhanced calibration with ");
  Serial.print(samples);
  Serial.println(" samples...");
  
  // Get baseline temperature
  calibration.baseline_temp = tempFilter.update(mpu.getTemperature() / 340.0 + 36.53);
  
  for (int i = 0; i < samples; i++) {
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);
    
    // Accumulate raw values for bias calculation
    ax_sum += ax_raw; ay_sum += ay_raw; az_sum += az_raw;
    gx_sum += gx_raw; gy_sum += gy_raw; gz_sum += gz_raw;
    
    // Also accumulate squares for variance calculation
    ax_sq_sum += (long)ax_raw * ax_raw;
    ay_sq_sum += (long)ay_raw * ay_raw;
    az_sq_sum += (long)az_raw * az_raw;
    gx_sq_sum += (long)gx_raw * gx_raw;
    gy_sq_sum += (long)gy_raw * gy_raw;
    gz_sq_sum += (long)gz_raw * gz_raw;
    
    // Progress indicator
    if (i % 100 == 0) {
      Serial.print("Progress: ");
      Serial.print((i * 100) / samples);
      Serial.println("%");
    }
    
    delay(5);
  }
  
  // Calculate biases (raw counts)
  calibration.ax_bias = (float)ax_sum / samples;
  calibration.ay_bias = (float)ay_sum / samples;
  calibration.az_bias = ((float)az_sum / samples) - 16384.0; // subtract 1g
  calibration.gx_bias = (float)gx_sum / samples;
  calibration.gy_bias = (float)gy_sum / samples;
  calibration.gz_bias = (float)gz_sum / samples;
  
  // Calculate variances (for quality assessment)
  calibration.ax_variance = ((float)ax_sq_sum / samples) - pow(calibration.ax_bias, 2);
  calibration.ay_variance = ((float)ay_sq_sum / samples) - pow(calibration.ay_bias, 2);
  calibration.az_variance = ((float)az_sq_sum / samples) - pow(calibration.az_bias + 16384.0, 2);
  calibration.gx_variance = ((float)gx_sq_sum / samples) - pow(calibration.gx_bias, 2);
  calibration.gy_variance = ((float)gy_sq_sum / samples) - pow(calibration.gy_bias, 2);
  calibration.gz_variance = ((float)gz_sq_sum / samples) - pow(calibration.gz_bias, 2);
  
  calibration.is_valid = true;
  
  Serial.println("Enhanced calibration completed!");
  Serial.print("Accel biases (raw): ");
  Serial.print(calibration.ax_bias, 2); Serial.print(", ");
  Serial.print(calibration.ay_bias, 2); Serial.print(", ");
  Serial.println(calibration.az_bias, 2);
  Serial.print("Gyro biases (raw): ");
  Serial.print(calibration.gx_bias, 2); Serial.print(", ");
  Serial.print(calibration.gy_bias, 2); Serial.print(", ");
  Serial.println(calibration.gz_bias, 2);
}

String getDataQualityStatus() {
  if (quality.total_samples == 0) return "INIT";
  
  float error_rate = (float)quality.invalid_samples / quality.total_samples;
  
  if (error_rate < 0.01) return "EXCELLENT";
  else if (error_rate < 0.05) return "GOOD";
  else if (error_rate < 0.1) return "FAIR";
  else return "POOR";
}

bool validateReading(float ax, float ay, float az, float gx, float gy, float gz) {
  // Check for reasonable ranges
  if (abs(ax) > MAX_REASONABLE_ACCEL || abs(ay) > MAX_REASONABLE_ACCEL || abs(az) > MAX_REASONABLE_ACCEL) {
    return false;
  }
  if (abs(gx) > MAX_REASONABLE_GYRO || abs(gy) > MAX_REASONABLE_GYRO || abs(gz) > MAX_REASONABLE_GYRO) {
    return false;
  }
  
  // Check for spikes (if not first reading)
  if (!quality.first_reading) {
    float ax_diff = abs(ax - quality.last_ax);
    float ay_diff = abs(ay - quality.last_ay);
    float az_diff = abs(az - quality.last_az);
    
    if (ax_diff > SPIKE_DETECTION_THRESHOLD || ay_diff > SPIKE_DETECTION_THRESHOLD || az_diff > SPIKE_DETECTION_THRESHOLD) {
      return false;
    }
  }
  
  return true;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Initialize MPU6050
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("ERROR: MPU6050 connection failed!");
    while (1) {
      Serial.println("Please check wiring and reset.");
      delay(5000);
    }
  }
  
  Serial.println("MPU6050 connected successfully!");
  
  // Configure sensor ranges
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);   // ±2g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);   // ±250 dps
  mpu.setDLPFMode(3);   // 44 Hz cutoff
  mpu.setRate(9);       // 1000 / (1+9) = 100 Hz
  
  // Let sensor settle
  delay(2000);
  
  Serial.println("\n=== ENHANCED MPU6050 CALIBRATION ===");
  Serial.println("Place sensor on a stable, level surface.");
  Serial.println("Ensure no vibrations or movement during calibration.");
  
  // Multi-stage calibration process
  Serial.println("\n--- Stage 1: Stability Check ---");
  if (!checkSensorStability()) {
    Serial.println("WARNING: Proceeding with unstable sensor. Results may be poor.");
  }
  
  Serial.println("\n--- Stage 2: Enhanced Calibration ---");
  performEnhancedCalibration();
  
  Serial.println("\n--- Stage 3: Verification ---");
  Serial.println("Taking 50 test readings...");
  
  float test_sum = 0;
  int valid_tests = 0;
  for (int i = 0; i < 50; i++) {
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);
    
    float ax = ((ax_raw - calibration.ax_bias) / 16384.0) * 9.81;
    float ay = ((ay_raw - calibration.ay_bias) / 16384.0) * 9.81;
    float az = ((az_raw - calibration.az_bias) / 16384.0) * 9.81;
    
    float total_accel = sqrt(ax*ax + ay*ay + az*az);
    test_sum += total_accel;
    valid_tests++;
    
    delay(20);
  }
  
  float avg_total_accel = test_sum / valid_tests;
  float gravity_error = abs(avg_total_accel - 9.81);
  
  Serial.print("Calibration verification: Average total acceleration = ");
  Serial.print(avg_total_accel, 3);
  Serial.print(" m/s² (error from 9.81: ");
  Serial.print(gravity_error, 3);
  Serial.println(" m/s²)");
  
  if (gravity_error < 0.5) {
    Serial.println("✓ Calibration EXCELLENT");
  } else if (gravity_error < 1.0) {
    Serial.println("✓ Calibration GOOD");
  } else {
    Serial.println("⚠ Calibration POOR - consider recalibrating");
  }
  
  Serial.println("\n=== Starting Data Stream ===");
  Serial.println("Format: timestamp_ms,ax,ay,az,gx,gy,gz,temperature,status");
  delay(1000);
}
void loop() {
  quality.total_samples++;
  
  // Get raw sensor data
  int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
  mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);
  
  // Get temperature
  float temperature = tempFilter.update(mpu.getTemperature() / 340.0 + 36.53);
  
  // Apply calibration
  float axc = (float)ax_raw - calibration.ax_bias;
  float ayc = (float)ay_raw - calibration.ay_bias;
  float azc = (float)az_raw - calibration.az_bias;
  float gxc = (float)gx_raw - calibration.gx_bias;
  float gyc = (float)gy_raw - calibration.gy_bias;
  float gzc = (float)gz_raw - calibration.gz_bias;
  
  // Temperature compensation (simple linear model)
  if (tempFilter.isReady()) {
    float temp_diff = temperature - calibration.baseline_temp;
    if (abs(temp_diff) > TEMP_DRIFT_THRESHOLD) {
      // Apply simple temperature compensation (you can make this more sophisticated)
      float temp_factor = 1.0 + (temp_diff * 0.001); // 0.1% per degree C
      axc *= temp_factor;
      ayc *= temp_factor;
      azc *= temp_factor;
    }
  }
  
  // Convert to engineering units
  const float ACCEL_SCALE = 16384.0;
  const float G = 9.81;
  const float GYRO_SCALE = 131.0;
  
  float ax_ms2 = (axc / ACCEL_SCALE) * G;
  float ay_ms2 = (ayc / ACCEL_SCALE) * G;
  float az_ms2 = (azc / ACCEL_SCALE) * G;
  float gx_dps = gxc / GYRO_SCALE;
  float gy_dps = gyc / GYRO_SCALE;
  float gz_dps = gzc / GYRO_SCALE;
  
  // Validate reading
  bool is_valid = validateReading(ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps);
  if (!is_valid) {
    quality.invalid_samples++;
  }
  
  // Update quality tracking
  if (!quality.first_reading) {
    quality.last_ax = ax_ms2;
    quality.last_ay = ay_ms2;
    quality.last_az = az_ms2;
    quality.last_gx = gx_dps;
    quality.last_gy = gy_dps;
    quality.last_gz = gz_dps;
  }
  quality.first_reading = false;
  
  // Get timestamp
  unsigned long ts = millis();
  
  // Generate status string
  String status = getDataQualityStatus();
  if (!is_valid) status += "_INVALID";
  if (abs(temperature - calibration.baseline_temp) > TEMP_DRIFT_THRESHOLD) {
    status += "_TEMP_DRIFT";
  }
  
  // Output CSV data
  Serial.print(ts); Serial.print(",");
  Serial.print(ax_ms2, 4); Serial.print(",");
  Serial.print(ay_ms2, 4); Serial.print(",");
  Serial.print(az_ms2, 4); Serial.print(",");
  Serial.print(gx_dps, 4); Serial.print(",");
  Serial.print(gy_dps, 4); Serial.print(",");
  Serial.print(gz_dps, 4); Serial.print(",");
  Serial.print(temperature, 2); Serial.print(",");
  Serial.println(status);
  
  // Maintain 100 Hz sample rate
  delay(10);
}