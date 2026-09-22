#include <Arduino.h>
#include <WiFi.h>
#include <esp_mac.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "img_converters.h"
#include "config.h"

// TensorFlow Lite Micro Headers
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// ==============================================================================
// 1. TFLITE MICRO SETUP
// ==============================================================================
constexpr int kTensorArenaSize = 128 * 1024; // 128 KB สำหรับโมเดลขนาดเล็ก
uint8_t* tensor_arena = nullptr;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// เรียงลำดับคลาสตามตัวอักษร A-Z ให้ตรงกับ TensorFlow image_dataset_from_directory
// Index 0: overripe, Index 1: ripe, Index 2: unripe
const char* class_names[] = {"overripe", "ripe", "unripe"};

struct InferenceResult {
  char best_class[16];
  float probabilities[3]; // [0]=overripe, [1]=ripe, [2]=unripe
  float prep_ms;
  float infer_ms;
  float total_ms;
  float fps;
};
InferenceResult latest_result = {"Waiting...", {0.0f, 0.0f, 0.0f}, 0, 0, 0, 0};

SemaphoreHandle_t dataMutex = NULL;
uint8_t* latest_jpg_buf = nullptr;
size_t latest_jpg_len = 0;

httpd_handle_t server_httpd = NULL;

// ==============================================================================
// 2. CAMERA INITIALIZATION (96x96 RGB565)
// ==============================================================================
bool initCamera() {
  pinMode(PWR_ON_PIN, OUTPUT);
  digitalWrite(PWR_ON_PIN, HIGH);
  delay(150);

#ifdef CAM_IR_PIN
  pinMode(CAM_IR_PIN, OUTPUT);
  digitalWrite(CAM_IR_PIN, LOW);
#endif

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = CAM_Y2_PIN;
  config.pin_d1 = CAM_Y3_PIN;
  config.pin_d2 = CAM_Y4_PIN;
  config.pin_d3 = CAM_Y5_PIN;
  config.pin_d4 = CAM_Y6_PIN;
  config.pin_d5 = CAM_Y7_PIN;
  config.pin_d6 = CAM_Y8_PIN;
  config.pin_d7 = CAM_Y9_PIN;

  config.pin_xclk = CAM_XCLK_PIN;
  config.pin_pclk = CAM_PCLK_PIN;
  config.pin_vsync = CAM_VSYNC_PIN;
  config.pin_href  = CAM_HREF_PIN;

  config.pin_sccb_sda = CAM_SIOD_PIN;
  config.pin_sccb_scl = CAM_SIOC_PIN;

  config.pin_pwdn  = CAM_PWDN_PIN;
  config.pin_reset = CAM_RESET_PIN;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_96X96;
  config.jpeg_quality = 12;
  config.fb_count     = 2;
  config.fb_location  = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    return false;
  }

  // ปรับแต่ง Sensor หลัง esp_camera_init สำเร็จแล้วเท่านั้น
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_whitebal(s, 1);       // เปิด Auto White Balance
    s->set_awb_gain(s, 1);       // เปิด AWB Gain
    s->set_wb_mode(s, 0);        // 0 = Auto WB Mode
    s->set_saturation(s, 1);     // เพิ่มความสดของสีเล็กน้อย (+1 จาก -2 ถึง 2)
    s->set_brightness(s, 0);     // ความสว่างระดับกลาง
    s->set_contrast(s, 0);       // คอนทราสต์ระดับกลาง
  }

  return true;
}

// ==============================================================================
// 3. PREPROCESSING (RGB565 -> ZERO-CENTERED INT8)
// ==============================================================================
void preprocessAndLoad(camera_fb_t* fb) {
  int8_t* input_buffer = input->data.int8;
  uint8_t* buf = fb->buf;

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  const float inv_factor = 1.0f / (127.5f * input_scale);

  int idx = 0;
  for (int i = 0; i < 96 * 96; i++) {
    // รวม 2 ไบต์ให้เป็น 1 พิกเซลแบบ Big-Endian ที่ถูกต้อง
    uint8_t byte1 = buf[i * 2];
    uint8_t byte2 = buf[i * 2 + 1];
    uint16_t pixel = (byte1 << 8) | byte2;

    int r = ((pixel >> 11) & 0x1F) << 3;
    int g = ((pixel >> 5)  & 0x3F) << 2;
    int b = (pixel & 0x1F) << 3;

    // Normalization & Quantization ไปเป็น int8
    input_buffer[idx++] = (int8_t)constrain((int)round((r - 127.5f) * inv_factor) + input_zero_point, -128, 127);
    input_buffer[idx++] = (int8_t)constrain((int)round((g - 127.5f) * inv_factor) + input_zero_point, -128, 127);
    input_buffer[idx++] = (int8_t)constrain((int)round((b - 127.5f) * inv_factor) + input_zero_point, -128, 127);
  }
}

// ==============================================================================
// 4. WEB SERVER (HTML, Snapshot, JSON Result)
// ==============================================================================
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>T-SIMCAM Mangosteen Classifier</title>
  <style>
    body { font-family: sans-serif; background: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
    .card { background: #1e1e1e; max-width: 400px; margin: auto; border-radius: 12px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    img { width: 192px; height: 192px; border-radius: 8px; border: 2px solid #333; image-rendering: pixelated; background: #000; }
    .pred { font-size: 1.5rem; font-weight: bold; color: #4CAF50; margin: 15px 0 5px 0; text-transform: uppercase; }
    .bar-wrap { background: #333; border-radius: 6px; overflow: hidden; height: 16px; margin: 4px 0 10px 0; }
    .bar { height: 100%; background: #2196F3; width: 0%; transition: width 0.2s; }
    .label { display: flex; justify-content: space-between; font-size: 0.85rem; color: #ccc; }
    .stats { font-size: 0.75rem; color: #777; margin-top: 15px; border-top: 1px solid #333; padding-top: 10px; }
  </style>
</head>
<body>
  <div class="card">
    <h3>Mangosteen Classifier</h3>
    <img id="stream" src="/capture" alt="Feed">
    <div id="pred" class="pred">Waiting...</div>

    <div class="label"><span>Unripe</span><span id="p-unripe">0%</span></div>
    <div class="bar-wrap"><div id="b-unripe" class="bar"></div></div>

    <div class="label"><span>Ripe</span><span id="p-ripe">0%</span></div>
    <div class="bar-wrap"><div id="b-ripe" class="bar"></div></div>

    <div class="label"><span>Overripe</span><span id="p-overripe">0%</span></div>
    <div class="bar-wrap"><div id="b-overripe" class="bar"></div></div>

    <div class="stats" id="stats">Ready</div>
  </div>

  <script>
    setInterval(() => {
      document.getElementById('stream').src = '/capture?t=' + Date.now();
      fetch('/result')
        .then(r => r.json())
        .then(d => {
          document.getElementById('pred').innerText = d.class;
          document.getElementById('p-unripe').innerText = d.unripe + '%';
          document.getElementById('b-unripe').style.width = d.unripe + '%';
          document.getElementById('p-ripe').innerText = d.ripe + '%';
          document.getElementById('b-ripe').style.width = d.ripe + '%';
          document.getElementById('p-overripe').innerText = d.overripe + '%';
          document.getElementById('b-overripe').style.width = d.overripe + '%';
          document.getElementById('stats').innerText = `Infer: ${d.infer_ms} ms | Total: ${d.total_ms} ms (${d.fps} FPS)`;
        }).catch(()=>{});
    }, 400);
  </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

static esp_err_t capture_handler(httpd_req_t *req) {
  esp_err_t res = ESP_FAIL;
  if (xSemaphoreTake(dataMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    if (latest_jpg_buf != nullptr && latest_jpg_len > 0) {
      httpd_resp_set_type(req, "image/jpeg");
      httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
      res = httpd_resp_send(req, (const char*)latest_jpg_buf, latest_jpg_len);
    } else {
      httpd_resp_send_500(req);
    }
    xSemaphoreGive(dataMutex);
  } else {
    httpd_resp_send_500(req);
  }
  return res;
}

static esp_err_t result_handler(httpd_req_t *req) {
  char json[256];
  if (xSemaphoreTake(dataMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    // Map ตรงกับ Index: [0]=overripe, [1]=ripe, [2]=unripe
    snprintf(json, sizeof(json),
      "{\"class\":\"%s\",\"overripe\":%.1f,\"ripe\":%.1f,\"unripe\":%.1f,\"infer_ms\":%.1f,\"total_ms\":%.1f,\"fps\":%.1f}",
      latest_result.best_class,
      latest_result.probabilities[0] * 100.0f,
      latest_result.probabilities[1] * 100.0f,
      latest_result.probabilities[2] * 100.0f,
      latest_result.infer_ms,
      latest_result.total_ms,
      latest_result.fps
    );
    xSemaphoreGive(dataMutex);
  } else {
    snprintf(json, sizeof(json), "{\"class\":\"Busy\"}");
  }

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json, strlen(json));
}

void startWebServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.stack_size = 8192;

  httpd_uri_t index_uri = { .uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL };
  httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
  httpd_uri_t result_uri = { .uri = "/result", .method = HTTP_GET, .handler = result_handler, .user_ctx = NULL };

  if (httpd_start(&server_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(server_httpd, &index_uri);
    httpd_register_uri_handler(server_httpd, &capture_uri);
    httpd_register_uri_handler(server_httpd, &result_uri);
  }
}

// ==============================================================================
// 5. SETUP
// ==============================================================================
void setup() {
  setCpuFrequencyMhz(240); // รัน CPU ที่ 240 MHz เต็มสปีด

  Serial.begin(115200);
  delay(1000);

  Serial.println("\n[INIT] Starting T-SIMCAM AI + Web Server...");
  dataMutex = xSemaphoreCreateMutex();

  // 1. Memory Arena
  if (psramFound()) {
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  } else {
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  }
  if (!tensor_arena) {
    Serial.println("[FATAL] Arena allocation failed!");
    while (true) delay(1000);
  }

  // 2. Camera
  if (!initCamera()) {
    Serial.println("[ERROR] Camera Init Failed!");
    while (true) delay(1000);
  }
  Serial.println("[OK] Camera Initialized.");

  // 3. TFLite Interpreter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(mangosteen_quant_int8_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[FATAL] AllocateTensors Failed!");
    while (true) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("[OK] TFLite Ready.");

  // 4. Wi-Fi SoftAP
  String ssid = WIFI_AP_SSID;
  uint8_t mac[8];
  esp_efuse_mac_get_default(mac);
  ssid += String(mac[0] + mac[1] + mac[2]);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid.c_str(), WIFI_AP_PASSWORD);

  // 5. Start Web Server
  startWebServer();

  Serial.println("==========================================");
  Serial.print("Wi-Fi SSID : "); Serial.println(ssid);
  Serial.print("Web UI     : http://"); Serial.println(WiFi.softAPIP());
  Serial.println("==========================================");
}

// ==============================================================================
// 6. MAIN LOOP
// ==============================================================================
void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    vTaskDelay(pdMS_TO_TICKS(30));
    return;
  }

  int64_t t_start_prep = esp_timer_get_time();
  preprocessAndLoad(fb);

  // แปลง RGB565 เป็น JPEG สำหรับหน้าเว็บ
  uint8_t* temp_jpg = nullptr;
  size_t temp_len = 0;
  bool jpg_ok = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_RGB565, 80, &temp_jpg, &temp_len);
  int64_t t_prep = esp_timer_get_time() - t_start_prep;

  esp_camera_fb_return(fb);

  // สลับ Buffer ภาพอย่างปลอดภัย
  if (jpg_ok) {
    uint8_t* old_buf = nullptr;
    if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
      old_buf = latest_jpg_buf;
      latest_jpg_buf = temp_jpg;
      latest_jpg_len = temp_len;
      xSemaphoreGive(dataMutex);
    }
    if (old_buf) {
      free(old_buf);
    }
  }

  // AI Inference
  int64_t t_start_infer = esp_timer_get_time();
  interpreter->Invoke();
  int64_t t_infer = esp_timer_get_time() - t_start_infer;

  // Dequantize ผลลัพธ์
  int8_t* out_data = output->data.int8;
  float out_scale = output->params.scale;
  int out_zero_point = output->params.zero_point;

  int best_idx = 0;
  float max_p = -100.0f;
  float probs[3];

  for (int i = 0; i < 3; i++) {
    probs[i] = (out_data[i] - out_zero_point) * out_scale;
    if (probs[i] > max_p) {
      max_p = probs[i];
      best_idx = i;
    }
  }

  // อัปเดตผลลัพธ์ลงตัวแปรกลางสำหรับ Web UI
  if (xSemaphoreTake(dataMutex, portMAX_DELAY) == pdTRUE) {
    strncpy(latest_result.best_class, class_names[best_idx], sizeof(latest_result.best_class) - 1);
    latest_result.probabilities[0] = probs[0]; // overripe
    latest_result.probabilities[1] = probs[1]; // ripe
    latest_result.probabilities[2] = probs[2]; // unripe
    latest_result.prep_ms = t_prep / 1000.0f;
    latest_result.infer_ms = t_infer / 1000.0f;
    latest_result.total_ms = (t_prep + t_infer) / 1000.0f;
    latest_result.fps = 1000.0f / latest_result.total_ms;
    xSemaphoreGive(dataMutex);
  }

  vTaskDelay(pdMS_TO_TICKS(30));
}