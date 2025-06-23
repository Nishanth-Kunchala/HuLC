const int pwmPins[6] = {3, 2, 7, 6, 5, 9};  // PWM-capable pins

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 6; i++) {
    pinMode(pwmPins[i], OUTPUT);
  }
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    int pwmValues[6] = {0};

    int i = 0;
    char *ptr = strtok(const_cast<char*>(input.c_str()), ",");
    while (ptr != NULL && i < 6) {
      pwmValues[i] = constrain(atoi(ptr), 0, 255);
      ptr = strtok(NULL, ",");
      i++;
    }

    for (int j = 0; j < 6; j++) {
      analogWrite(pwmPins[j], pwmValues[j]);
    }

    Serial.print("Set PWM: ");
    for (int j = 0; j < 6; j++) {
      Serial.print(pwmValues[j]);
      Serial.print(j < 5 ? "," : "\n");
    }
  }
}
