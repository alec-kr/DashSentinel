#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// common i2c address for ssd1306 oleds
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String currentStatus = "NO DATA";
float attentiveness = 0.0;
float drowsyScore = 0.0;

void drawDisplay() {
  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("DashSentinel");

  display.setCursor(0, 14);
  display.print("Status:");

  display.setTextSize(2);
  display.setCursor(0, 24);
  display.print(currentStatus);

  display.setTextSize(1);
  display.setCursor(0, 48);
  display.print("Attention: ");
  display.print(attentiveness, 1);
  display.print("%");

  display.setCursor(0, 57);
  display.print("Drowsy: ");
  display.print(drowsyScore, 2);

  display.display();
}

void parseLine(String line) {
  line.trim();

  int firstComma = line.indexOf(',');
  int secondComma = line.indexOf(',', firstComma + 1);

  if (firstComma < 0 || secondComma < 0) {
    return;
  }

  currentStatus = line.substring(0, firstComma);
  attentiveness = line.substring(firstComma + 1, secondComma).toFloat();
  drowsyScore = line.substring(secondComma + 1).toFloat();

  drawDisplay();
}

void setup() {
  Serial.begin(115200);

  Wire.begin(); 
  // for many esp8266 boards: sda = d2/gpio4, scl = d1/gpio5
  // if needed, use: Wire.begin(D2, D1);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    while (true) {
      delay(1000);
    }
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("DashSentinel");
  display.println("waiting for data...");
  display.display();
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    parseLine(line);
  }
}