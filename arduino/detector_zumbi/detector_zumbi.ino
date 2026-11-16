// Maquete: servo movimenta somente um mecanismo simbolico.
#include <Servo.h>

const bool TESTE_MANUAL = false; // true para testar Z/N pelo Monitor Serial.
const byte TRIG = 11, ECHO = 9, PINO_SERVO = 7;
const byte PINO_SERVO_VARREDURA = 13;
const int VARREDURA_MIN = 20, VARREDURA_MAX = 160;
const unsigned long PASSO_VARREDURA_MS = 20;
const float DISTANCIA_CM = 200.0;
const int REPOUSO = 0, ACIONADO = 60;
const unsigned long TEMPO_RESPOSTA = 1500, TEMPO_PULSO = 1000, INTERVALO = 1500;
Servo servo;
Servo servoVarredura;
int anguloVarredura = 90, sentidoVarredura = 1;
unsigned long ultimoPassoVarredura = 0;
unsigned long ultimaMedicao = 0, pedidoEm = 0, pulsoEm = 0, ultimoPedido = 0;
bool perto = false, aguardando = false, ativo = false;
bool respondido = false;
char comando[8];
byte tamanho = 0;

void repousar() {
  servo.write(REPOUSO);
  ativo = false;
}

void setup() {
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  servo.attach(PINO_SERVO);
  servoVarredura.attach(PINO_SERVO_VARREDURA);
  servoVarredura.write(anguloVarredura);
  repousar();
  Serial.begin(9600);
  if (TESTE_MANUAL) Serial.println("Teste: aproxime um objeto a menos de 200 cm.");
}

void loop() {
  unsigned long agora = millis();
  if (agora - ultimaMedicao >= 100) {
    ultimaMedicao = agora;
    digitalWrite(TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG, LOW);
    unsigned long duracao = pulseIn(ECHO, HIGH, 25000UL);
    bool anterior = perto;
    perto = duracao > 0 && duracao * 0.0343 / 2.0 < DISTANCIA_CM;
    if (!perto) {
      respondido = false;
      aguardando = false;
      repousar();
      if (anterior) Serial.println("LONGE");
    }
  }
  agora = millis();
  // Sem delay: continua lendo o sensor e os comandos durante a varredura.
  if (!perto && agora - ultimoPassoVarredura >= PASSO_VARREDURA_MS) {
    ultimoPassoVarredura = agora;
    anguloVarredura += sentidoVarredura;
    if (anguloVarredura >= VARREDURA_MAX || anguloVarredura <= VARREDURA_MIN) {
      sentidoVarredura = -sentidoVarredura;
    }
    servoVarredura.write(anguloVarredura);
  }
  if (ativo && agora - pulsoEm >= TEMPO_PULSO) repousar();
  if (!TESTE_MANUAL && aguardando && agora - pedidoEm >= TEMPO_RESPOSTA) {
    aguardando = false;
    repousar();
  }
  while (Serial.available()) {
    char c = Serial.read();
    if (TESTE_MANUAL) {
      if (aguardando && perto && (c == 'Z' || c == 'z' || c == 'N' || c == 'n')) {
        aguardando = false;
        respondido = true;
        if (c == 'Z' || c == 'z') {
          ativo = true;
          pulsoEm = millis();
          servo.write(ACIONADO);
          Serial.println("Z: servo acionado. Afaste e aproxime para repetir.");
        } else {
          repousar();
          Serial.println("N: servo em repouso. Afaste e aproxime para repetir.");
        }
      }
      continue;
    }
    if (c == '\n') {
      comando[tamanho] = '\0';
      if (strcmp(comando, "N") == 0) {
        aguardando = false;
        repousar();
      } else if (strcmp(comando, "Z") == 0 && aguardando && perto &&
                 millis() - pedidoEm < TEMPO_RESPOSTA) {
        aguardando = false;
        ativo = true;
        pulsoEm = millis();
        servo.write(ACIONADO);
      }
      tamanho = 0;
    } else if (c != '\r') {
      if (tamanho < sizeof(comando) - 1) comando[tamanho++] = c;
    }
  }
  if (perto && !aguardando && !ativo && (!TESTE_MANUAL || !respondido) &&
      agora - ultimoPedido >= INTERVALO) {
    Serial.println("ANALISAR");
    if (TESTE_MANUAL) Serial.println("Envie Z para girar o servo ou N para manter em repouso.");
    aguardando = true;
    pedidoEm = ultimoPedido = millis();
  }
}
