#include "DHT.h"
#include <ArduinoJson.h>

#define DHTPIN PA6
#define doorPin PB0
#define gear PA7
#define LedP PA6
#define LedR PA5
#define LedN PA4
#define LedD PA3


#define DHTTYPE DHT11   // DHT 11
String gearSelected = "P";
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  pinMode(gear, INPUT);
  pinMode(doorPin, INPUT_PULLUP);
  dht.begin();
}

void loop() {
  int d = digitalRead(doorPin);
  int h = dht.readHumidity();
  int t = dht.readTemperature();  

  // Create a JSON object
  StaticJsonDocument<200> doc;

  // Check if any reads failed and exit early (to try again).
  if (isnan(h) || isnan(t)) {
    doc["temperature"] = String("--");
    doc["humidity"] = "N/A";
    doc["door"] = 1;
  } else {
    doc["temperature"] = t;
    doc["humidity"] = h;
    doc["door"] = d;
  }

  // Determine gear
  int gearValue = analogRead(gear);
  if (gearValue > 50 && gearValue < 800) { 
    gearSelected = "D";
  } else if (gearValue > 2000 && gearValue < 2070){
    gearSelected = "N";
  } else if (gearValue > 2700 && gearValue < 2960) {
    gearSelected = "R";
  } else if (gearValue > 3075 && gearValue < 4000) {
    gearSelected = "P";
  }
//  Serial.println(gearSelected);
//  Serial.println(gearValue);
  doc["gear"] = gearSelected;

  // Serialize JSON to a string and send it over serial
  serializeJson(doc, Serial);
  Serial.println();
  delay(150);
}
