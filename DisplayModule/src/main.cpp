#include <Arduino.h>
#include <Wire.h>
//Required for display
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// init OLED dimensions
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// define pins for buttons
#define BTN_RESET_BASELINE D5 // to reset the baseline for drowsiness detection
#define BTN_RESET_STATS D6 // to reset the drowsiness score and attentiveness score

// i2c address for ssd1306
#define OLED_ADDR 0x3C

// display object
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// store current status and scores
String currentStatus = "NO DATA";
float attentiveness = 0.0;
float drowsyScore = 0.0;

// store last button states
bool lastBaselineState = HIGH;
bool lastStatBtnState = HIGH;

// store last button press times for debouncing
unsigned long lastBaselinePress = 0;
unsigned long lastStatBtnPress = 0;
const unsigned long debounceMs = 250;

// show current status and scores on the display
void drawDisplay() {
  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("DashSentinel");

  display.setCursor(0, 16);
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

// parse incoming serial data in format "status,attentiveness,drowsyScore"
void parseLine(String line) {
  line.trim(); // remove whitespace and newline chars

  // split by comma
  int firstComma = line.indexOf(',');
  int secondComma = line.indexOf(',', firstComma + 1);

  // validate format. if comma is missing, ignore the line
  if (firstComma < 0 || secondComma < 0) {
    return;
  }

  // extract vals from line
  currentStatus = line.substring(0, firstComma);
  attentiveness = line.substring(firstComma + 1, secondComma).toFloat();
  drowsyScore = line.substring(secondComma + 1).toFloat();

  // update display
  drawDisplay();
}

// Check button states and print to serial if pressed
void checkButtons() {
  // get current btn states
  bool baselineState = digitalRead(BTN_RESET_BASELINE);
  bool statBtnState = digitalRead(BTN_RESET_STATS);
  // used for debouncing
  unsigned long now = millis();

  // baseline reset button (D5)
  if (lastBaselineState == HIGH && baselineState == LOW && (now - lastBaselinePress > debounceMs)) {
    Serial.println("BTN_RESET_BASELINE");
    lastBaselinePress = now;
  }

  // stats reset button (D6)
  if (lastStatBtnState == HIGH && statBtnState == LOW && (now - lastStatBtnPress > debounceMs)) {
    Serial.println("BTN_RESET_STATS");
    lastStatBtnPress = now;
  }

  // update last states
  lastBaselineState = baselineState;
  lastStatBtnState = statBtnState;
}

void setup() {
  // using internal pullup resistors for buttons, 
  // so they read HIGH when not pressed and LOW when pressed
  pinMode(BTN_RESET_BASELINE, INPUT_PULLUP);
  pinMode(BTN_RESET_STATS, INPUT_PULLUP);

  Serial.begin(115200);
  Wire.begin(); 

  // delay to allow serial monitor to connect before sending data
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    while (true) {
      delay(1000);
    }
  }

  // base init for display
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("DashSentinel");
  display.println("waiting for data...");
  display.display();
}

void loop() {
  // check buttons on every loop and print to serial if pressed
  checkButtons();

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    parseLine(line);
  }
}