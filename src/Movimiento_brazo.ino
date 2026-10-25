#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Dirección I2C predeterminada del PCA9685.
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const uint8_t NUM_SERVOS = 6;

// =====================================================
// INDICADOR FISICO LED RGB
// =====================================================
//
// Se utilizan solo los canales ROJO y VERDE:
//   OK  -> verde
//   NOK -> rojo
//   OFF -> apagado
//
// Conexion recomendada para LED RGB de catodo comun:
//   D5 -> resistencia 220-330 ohm -> ROJO
//   D6 -> resistencia 220-330 ohm -> VERDE
//   COMUN -> GND
//
// Si tu LED es de ANODO COMUN:
//   COMUN -> 5V
//   cambia LED_RGB_ANODO_COMUN a true.
const uint8_t PIN_LED_ROJO = 5;
const uint8_t PIN_LED_VERDE = 6;
const bool LED_RGB_ANODO_COMUN = false;

enum EstadoLedRgb {
  LED_ESTADO_OFF,
  LED_ESTADO_OK,
  LED_ESTADO_NOK
};

EstadoLedRgb estadoLedRgb = LED_ESTADO_OFF;

// =====================================================
// IDENTIFICACIÓN DE LOS SERVOS
// =====================================================

enum ServoID {
  BASE = 0,
  HOMBRO = 1,
  CODO = 2,
  MUNECA_PITCH = 3,
  MUNECA_ROLL = 4,
  PINZA = 5
};

enum EstadoBrazo {
  ESTADO_HOME,
  ESTADO_DROP_OK,
  ESTADO_DROP_NOK,
  ESTADO_LIFT,
  ESTADO_DESCONOCIDO
};

EstadoBrazo estadoBrazo = ESTADO_HOME;

// =====================================================
// CONFIGURACIÓN DE CADA SERVO
// =====================================================

struct ConfigServo {
  uint8_t canal;
  int pulsoMin;
  int pulsoMax;
  int pulsoHome;
  const char* nombre;
};

// Formato:
// {canal, pulso mínimo, pulso máximo, HOME, nombre}
ConfigServo servos[NUM_SERVOS] = {
  {0,  520, 2240, 1000, "Base"},
  {1, 1500, 2480, 1500, "Hombro"},
  {2,  520, 1500,  520, "Codo"},
  {3,  520, 2000, 2000, "Muneca/Pitch"},
  {4,  520, 2500, 1500, "Muneca/Roll"},
  {5,  520, 1280,  520, "Pinza"}
};

// El programa supone que físicamente el brazo comienza en HOME.
int posicionActual[NUM_SERVOS] = {
  1000,  // Base
  1500,  // Hombro
  520,   // Codo
  2000,  // Muñeca/Pitch
  1500,  // Muñeca/Roll
  520    // Pinza abierta
};

bool servosActivados = false;
bool brazoOcupado = false;

// =====================================================
// AJUSTES DE MOVIMIENTO
// =====================================================

// Menor paso y mayor retardo producen movimientos más lentos.
const int PASO_US = 5;
const int RETARDO_MS = 12;

// Tiempo para que el brazo se estabilice entre poses.
const int PAUSA_POSE_MS = 700;

// Posiciones de la pinza.
const int PINZA_ABIERTA = 520;
const int PINZA_CERRADA = 770;

// =====================================================
// POSES DE RECOGIDA
// =====================================================

const int POSE_PRE_PICK[NUM_SERVOS] = {
  1452,  // Base
  1500,  // Hombro
  520,   // Codo
  1630,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  520    // Pinza abierta
};

const int POSE_PICK[NUM_SERVOS] = {
  1452,  // Base
  1500,  // Hombro
  1300,  // Codo
  1660,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  520    // Pinza abierta
};

const int POSE_LIFT[NUM_SERVOS] = {
  1452,  // Base
  1500,  // Hombro
  1100,  // Codo
  1740,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  770    // Pinza cerrada
};

// =====================================================
// POSES DE TRANSPORTE Y CLASIFICACIÓN
// =====================================================

const int POSE_TRANSPORTE_SEGURO[NUM_SERVOS] = {
  1468,  // Base
  1500,  // Hombro
  700,   // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  770    // Pinza cerrada
};

const int POSE_PRE_DROP_OK[NUM_SERVOS] = {
  520,   // Base
  1500,  // Hombro
  700,   // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  770    // Pinza cerrada
};

const int POSE_DROP_OK[NUM_SERVOS] = {
  520,   // Base
  1500,  // Hombro
  1000,  // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  770    // Pinza cerrada
};

const int POSE_PRE_DROP_NOK[NUM_SERVOS] = {
  2240,   // Base
  1500,  // Hombro
  700,   // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  770    // Pinza cerrada
};

const int POSE_DROP_NOK[NUM_SERVOS] = {
  2240,   // Base
  1500,  // Hombro
  1000,  // Codo
  2000,  // Muñeca/Pitch
  1250,  // Muñeca/Roll
  770    // Pinza cerrada
};

// Retirada del contenedor OK con la pinza abierta.
const int POSE_RETIRO_DROP_OK[NUM_SERVOS] = {
  520,   // Base
  1500,  // Hombro
  1000,   // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  520    // Pinza abierta
};

// Retirada del contenedor NOK con la pinza abierta.
const int POSE_RETIRO_DROP_NOK[NUM_SERVOS] = {
  2240,   // Base
  1500,  // Hombro
  1000,   // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  520    // Pinza abierta
};

// Posición de transporte sin pieza.
const int POSE_TRANSPORTE_VACIO[NUM_SERVOS] = {
  1468,  // Base
  1500,  // Hombro
  700,   // Codo
  2000,  // Muñeca/Pitch
  1440,  // Muñeca/Roll
  520    // Pinza abierta
};

// =====================================================
// DECLARACIONES DE FUNCIONES
// =====================================================

void desactivarServos();
void activarEnHome();
void irHomeSeguro();

void escribirServo(uint8_t idServo, int pulso);
void moverServoSuave(uint8_t idServo, int destino);
void moverServoManual(uint8_t idServo, int destino);
void moverPoseSuave(const int destino[NUM_SERVOS]);

void secuenciaRecoger();

void mostrarEstado();
void mostrarAyuda();
void mostrarPose(const int pose[NUM_SERVOS], const char* nombre);

void secuenciaClasificarOK();
void secuenciaClasificarNOK();
bool prepararParaNuevaRecogida();
bool recogerPieza();

void inicializarLedRgb();
void aplicarLedRgb(bool rojoEncendido, bool verdeEncendido);
void ledRgbOff();
void ledRgbOk();
void ledRgbNok();
bool procesarComandoLedRgb(const String& comando);
const char* nombreEstadoLedRgb();

// =====================================================
// CONFIGURACIÓN INICIAL
// =====================================================

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);

  // El indicador inicia apagado.
  inicializarLedRgb();

  Wire.begin();

  pwm.begin();
  pwm.setPWMFreq(50);

  delay(500);

  // Arranca sin enviar señal a los servos.
  desactivarServos();

  Serial.println();
  Serial.println(F("======================================"));
  Serial.println(F(" CONTROL DE BRAZO ROBOTICO - PCA9685"));
  Serial.println(F("======================================"));
  Serial.println(F("El brazo inicia con los servos apagados."));
  Serial.println(F("Coloca fisicamente el brazo en HOME."));
  Serial.println(F("Luego escribe A para activar los servos."));
  Serial.println();

  mostrarAyuda();
}

// =====================================================
// CICLO PRINCIPAL
// =====================================================

void loop() {
  if (!Serial.available()) {
    return;
  }

  String comando = Serial.readStringUntil('\n');
  comando.trim();
  comando.toUpperCase();

  if (comando.length() == 0) {
    return;
  }

  // Los comandos del LED son independientes de las trayectorias del brazo.
  // Python espera la respuesta exacta LED_DONE.
  if (procesarComandoLedRgb(comando)) {
    return;
  }

  // Activar servos en la posición HOME.
  if (comando == "A") {
    activarEnHome();
  }

  // Ejecutar regreso seguro a HOME.
  else if (comando == "H") {
    irHomeSeguro();
  }

  // Ejecutar recogida automática.
  else if (comando == "R") {
    secuenciaRecoger();
  }

  else if (comando == "OK") {
    secuenciaClasificarOK();
  }

  else if (comando == "NOK") {
    secuenciaClasificarNOK();
  }

  // Desactivar todos los servos.
  else if (comando == "X") {
    desactivarServos();
  }

  // Mostrar estado actual.
  else if (comando == "?") {
    mostrarEstado();
  }

  // Mostrar poses programadas.
  else if (comando == "POSES") {
    mostrarPose(POSE_PRE_PICK, "PRE_PICK");
    mostrarPose(POSE_PICK, "PICK");
    mostrarPose(POSE_LIFT, "LIFT");
  }

  // Mostrar ayuda.
  else if (comando == "AYUDA") {
    mostrarAyuda();
  }

  // Movimiento manual.
  // Formato: M<servo>,<pulso>
  // Ejemplo: M0,1900
  else if (comando.startsWith("M")) {
    int separador = comando.indexOf(',');

    if (separador == -1) {
      Serial.println(
        F("ERROR: usa M<servo>,<pulso>. Ejemplo: M0,1900")
      );
      return;
    }

    String textoServo = comando.substring(1, separador);
    String textoPulso = comando.substring(separador + 1);

    if (textoServo.length() == 0 || textoPulso.length() == 0) {
      Serial.println(
        F("ERROR: comando incompleto. Ejemplo: M0,1900")
      );
      return;
    }

    int idServo = textoServo.toInt();
    int pulso = textoPulso.toInt();

    moverServoManual(idServo, pulso);
  }

  else {
    Serial.println(F("Comando no reconocido."));
    mostrarAyuda();
  }
}

// =====================================================
// ESCRITURA DIRECTA DE UN SERVO
// =====================================================

void escribirServo(uint8_t idServo, int pulso) {
  if (idServo >= NUM_SERVOS) {
    return;
  }

  pulso = constrain(
    pulso,
    servos[idServo].pulsoMin,
    servos[idServo].pulsoMax
  );

  pwm.writeMicroseconds(
    servos[idServo].canal,
    pulso
  );

  posicionActual[idServo] = pulso;
}

// =====================================================
// MOVIMIENTO SUAVE DE UN SOLO SERVO
// =====================================================

void moverServoSuave(uint8_t idServo, int destino) {
  if (!servosActivados) {
    Serial.println(F("ERROR: primero activa los servos con A."));
    return;
  }

  if (idServo >= NUM_SERVOS) {
    Serial.println(F("ERROR: servo inexistente."));
    return;
  }

  destino = constrain(
    destino,
    servos[idServo].pulsoMin,
    servos[idServo].pulsoMax
  );

  int actual = posicionActual[idServo];

  Serial.print(F("Moviendo "));
  Serial.print(servos[idServo].nombre);
  Serial.print(F(": "));
  Serial.print(actual);
  Serial.print(F(" -> "));
  Serial.println(destino);

  if (actual == destino) {
    escribirServo(idServo, destino);
    return;
  }

  int direccion = destino > actual ? 1 : -1;

  while (actual != destino) {
    actual += direccion * PASO_US;

    if (
      (direccion > 0 && actual > destino) ||
      (direccion < 0 && actual < destino)
    ) {
      actual = destino;
    }

    escribirServo(idServo, actual);
    delay(RETARDO_MS);
  }
}

// =====================================================
// MOVIMIENTO MANUAL DE UN SERVO
// =====================================================

void moverServoManual(uint8_t idServo, int destino) {
  if (!servosActivados) {
    Serial.println(F("ERROR: activa los servos con A."));
    return;
  }

  if (brazoOcupado) {
    Serial.println(
      F("ERROR: el brazo esta ejecutando una secuencia.")
    );
    return;
  }

  if (idServo >= NUM_SERVOS) {
    Serial.println(F("ERROR: servo inexistente."));
    return;
  }

  if (
    destino < servos[idServo].pulsoMin ||
    destino > servos[idServo].pulsoMax
  ) {
    Serial.print(F("AVISO: el pulso fue limitado al rango "));
    Serial.print(servos[idServo].pulsoMin);
    Serial.print(F("-"));
    Serial.print(servos[idServo].pulsoMax);
    Serial.println(F(" us."));
  }

  moverServoSuave(idServo, destino);
  estadoBrazo = ESTADO_DESCONOCIDO;
}

// =====================================================
// MOVIMIENTO SIMULTÁNEO HACIA UNA POSE
// =====================================================

void moverPoseSuave(const int destino[NUM_SERVOS]) {
  if (!servosActivados) {
    Serial.println(F("ERROR: primero activa los servos con A."));
    return;
  }

  bool movimientoTerminado = false;

  while (!movimientoTerminado) {
    movimientoTerminado = true;

    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
      int destinoSeguro = constrain(
        destino[i],
        servos[i].pulsoMin,
        servos[i].pulsoMax
      );

      int actual = posicionActual[i];

      if (actual < destinoSeguro) {
        actual += PASO_US;

        if (actual > destinoSeguro) {
          actual = destinoSeguro;
        }

        movimientoTerminado = false;
      }

      else if (actual > destinoSeguro) {
        actual -= PASO_US;

        if (actual < destinoSeguro) {
          actual = destinoSeguro;
        }

        movimientoTerminado = false;
      }

      escribirServo(i, actual);
    }

    delay(RETARDO_MS);
  }
}

// =====================================================
// ACTIVAR SERVOS EN HOME
// =====================================================

void activarEnHome() {
  if (servosActivados) {
    Serial.println(F("Los servos ya estan activados."));
    return;
  }

  Serial.println(F("Activando servos en HOME..."));
  Serial.println(
    F("ADVERTENCIA: el brazo debe estar fisicamente en HOME.")
  );

  // Se activan en un orden seguro.
  escribirServo(PINZA, servos[PINZA].pulsoHome);
  delay(200);

  escribirServo(CODO, servos[CODO].pulsoHome);
  delay(200);

  escribirServo(MUNECA_ROLL, servos[MUNECA_ROLL].pulsoHome);
  delay(200);

  escribirServo(
    MUNECA_PITCH,
    servos[MUNECA_PITCH].pulsoHome
  );
  delay(200);

  escribirServo(HOMBRO, servos[HOMBRO].pulsoHome);
  delay(200);

  escribirServo(BASE, servos[BASE].pulsoHome);
  delay(200);

  estadoBrazo = ESTADO_HOME;
  servosActivados = true;

  Serial.println(F("READY"));
  Serial.println(F("Servos activados correctamente."));
}

// =====================================================
// REGRESO SEGURO A HOME
// =====================================================

void irHomeSeguro() {
  if (!servosActivados) {
    Serial.println(F("ERROR: primero activa los servos con A."));
    return;
  }

  if (brazoOcupado) {
    Serial.println(
      F("ERROR: el brazo ya esta ejecutando una secuencia.")
    );
    return;
  }

  brazoOcupado = true;

  Serial.println(F("BUSY"));
  Serial.println(F("Iniciando secuencia HOME segura..."));

  // 1. Abrir la pinza.
  moverServoSuave(
    PINZA,
    servos[PINZA].pulsoHome
  );

  // 2. Llevar el codo a HOME.
  moverServoSuave(
    CODO,
    servos[CODO].pulsoHome
  );

  // 3. Llevar Muñeca/Pitch a HOME.
  moverServoSuave(
    MUNECA_PITCH,
    servos[MUNECA_PITCH].pulsoHome
  );

  // 4. Centrar Muñeca/Roll.
  moverServoSuave(
    MUNECA_ROLL,
    servos[MUNECA_ROLL].pulsoHome
  );

  // 5. Con el brazo recogido, mover el hombro.
  moverServoSuave(
    HOMBRO,
    servos[HOMBRO].pulsoHome
  );

  // 6. Finalmente, girar la base.
  moverServoSuave(
    BASE,
    servos[BASE].pulsoHome
  );

  estadoBrazo = ESTADO_HOME;
  brazoOcupado = false;

  Serial.println(F("Secuencia HOME terminada."));
  Serial.println(F("DONE"));
}

// =====================================================
// SECUENCIA AUTOMÁTICA DE RECOGIDA
// =====================================================

void secuenciaRecoger() {
  if (!servosActivados) {
    Serial.println(F("ERROR: primero activa los servos con A."));
    return;
  }

  if (brazoOcupado) {
    Serial.println(
      F("ERROR: el brazo ya esta ejecutando una secuencia.")
    );
    return;
  }

  brazoOcupado = true;

  Serial.println(F("BUSY"));
  Serial.println(F("Iniciando secuencia de recogida..."));

  // 1. Abrir la pinza.
  Serial.println(F("Abriendo pinza..."));

  moverServoSuave(
    PINZA,
    PINZA_ABIERTA
  );

  delay(500);

  // 2. Moverse sobre la pieza.
  Serial.println(F("Moviendo a PRE_PICK..."));

  moverPoseSuave(
    POSE_PRE_PICK
  );

  delay(PAUSA_POSE_MS);

  // 3. Bajar hasta la posición de agarre.
  Serial.println(F("Moviendo a PICK..."));

  moverPoseSuave(
    POSE_PICK
  );

  delay(PAUSA_POSE_MS);

  // 4. Cerrar la pinza.
  Serial.println(F("Cerrando pinza..."));

  moverServoSuave(
    PINZA,
    PINZA_CERRADA
  );

  delay(1000);

  // 5. Elevar la pieza.
  Serial.println(F("Moviendo a LIFT..."));

  moverPoseSuave(
    POSE_LIFT
  );

  delay(1000);

  brazoOcupado = false;

  Serial.println(F("Secuencia de recogida terminada."));
  Serial.println(F("DONE"));
}

// =====================================================
// Separar la recogida en una función reutilizable
// =====================================================
bool recogerPieza() {
  // Si está en DROP, primero sale del contenedor.
  if (!prepararParaNuevaRecogida()) {
    return false;
  }

  Serial.println(F("Abriendo pinza..."));
  moverServoSuave(PINZA, PINZA_ABIERTA);
  delay(500);

  Serial.println(F("Moviendo a PRE_PICK..."));
  moverPoseSuave(POSE_PRE_PICK);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Moviendo a PICK..."));
  moverPoseSuave(POSE_PICK);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Cerrando pinza..."));
  moverServoSuave(PINZA, PINZA_CERRADA);
  delay(1000);

  Serial.println(F("Elevando pieza a LIFT..."));
  moverPoseSuave(POSE_LIFT);
  delay(1000);

  estadoBrazo = ESTADO_LIFT;

  return true;
}

// =====================================================
// Secuancia para piezas OK
// =====================================================
void secuenciaClasificarOK() {
  if (!servosActivados) {
    Serial.println(
      F("ERROR: primero activa los servos con A.")
    );
    return;
  }

  if (brazoOcupado) {
    Serial.println(
      F("ERROR: el brazo ya esta ocupado.")
    );
    return;
  }

  brazoOcupado = true;

  Serial.println(F("BUSY"));
  Serial.println(F("CLASIFICACION: OK"));

  // Recoger la nueva pieza.
  if (!recogerPieza()) {
    brazoOcupado = false;
    Serial.println(F("ERROR_RECOGIDA"));
    return;
  }

  Serial.println(F("Moviendo a TRANSPORTE_SEGURO..."));
  moverPoseSuave(POSE_TRANSPORTE_SEGURO);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Moviendo a PRE_DROP_OK..."));
  moverPoseSuave(POSE_PRE_DROP_OK);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Moviendo a DROP_OK..."));
  moverPoseSuave(POSE_DROP_OK);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Abriendo pinza..."));
  moverServoSuave(PINZA, PINZA_ABIERTA);
  delay(1000);

  // El brazo permanece en DROP_OK.
  estadoBrazo = ESTADO_DROP_OK;
  brazoOcupado = false;

  Serial.println(F("Pieza depositada en OK."));
  Serial.println(F("Brazo esperando en DROP_OK."));
  Serial.println(F("DONE"));
}
// =====================================================
// Secuencia para piezas NOK
// =====================================================
void secuenciaClasificarNOK() {
  if (!servosActivados) {
    Serial.println(
      F("ERROR: primero activa los servos con A.")
    );
    return;
  }

  if (brazoOcupado) {
    Serial.println(
      F("ERROR: el brazo ya esta ocupado.")
    );
    return;
  }

  brazoOcupado = true;

  Serial.println(F("BUSY"));
  Serial.println(F("CLASIFICACION: NOK"));

  // Recoger la nueva pieza.
  if (!recogerPieza()) {
    brazoOcupado = false;
    Serial.println(F("ERROR_RECOGIDA"));
    return;
  }

  Serial.println(F("Moviendo a TRANSPORTE_SEGURO..."));
  moverPoseSuave(POSE_TRANSPORTE_SEGURO);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Moviendo a PRE_DROP_NOK..."));
  moverPoseSuave(POSE_PRE_DROP_NOK);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Moviendo a DROP_NOK..."));
  moverPoseSuave(POSE_DROP_NOK);
  delay(PAUSA_POSE_MS);

  Serial.println(F("Abriendo pinza..."));
  moverServoSuave(PINZA, PINZA_ABIERTA);
  delay(1000);

  // El brazo permanece en DROP_NOK.
  estadoBrazo = ESTADO_DROP_NOK;
  brazoOcupado = false;

  Serial.println(F("Pieza depositada en NOK."));
  Serial.println(F("Brazo esperando en DROP_NOK."));
  Serial.println(F("DONE"));
}


bool prepararParaNuevaRecogida() {
  switch (estadoBrazo) {
    case ESTADO_HOME:
      Serial.println(F("El brazo inicia desde HOME."));
      return true;

    case ESTADO_DROP_OK:
      Serial.println(
        F("Retirando brazo del contenedor OK...")
      );

      // Elevar primero sin cerrar la pinza.
      moverPoseSuave(POSE_RETIRO_DROP_OK);
      delay(PAUSA_POSE_MS);

      // Regresar al punto central seguro.
      moverPoseSuave(POSE_TRANSPORTE_VACIO);
      delay(PAUSA_POSE_MS);

      return true;

    case ESTADO_DROP_NOK:
      Serial.println(
        F("Retirando brazo del contenedor NOK...")
      );

      // Elevar primero sin cerrar la pinza.
      moverPoseSuave(POSE_RETIRO_DROP_NOK);
      delay(PAUSA_POSE_MS);

      // Regresar al punto central seguro.
      moverPoseSuave(POSE_TRANSPORTE_VACIO);
      delay(PAUSA_POSE_MS);

      return true;

    case ESTADO_LIFT:
      Serial.println(
        F("ERROR: el brazo conserva una pieza en LIFT.")
      );
      Serial.println(
        F("Ejecuta H antes de iniciar otra clasificacion.")
      );
      return false;

    case ESTADO_DESCONOCIDO:
    default:
      Serial.println(
        F("ERROR: posicion del brazo desconocida.")
      );
      Serial.println(
        F("Ejecuta H para recuperar una posicion segura.")
      );
      return false;
  }
}

// =====================================================
// DESACTIVAR TODOS LOS SERVOS
// =====================================================

void desactivarServos() {
  for (uint8_t canal = 0; canal < 16; canal++) {
    pwm.setPWM(canal, 0, 4096);
  }

  servosActivados = false;
  brazoOcupado = false;

  Serial.println(F("Todos los servos fueron desactivados."));
}

// =====================================================
// MOSTRAR ESTADO
// =====================================================

void mostrarEstado() {
  Serial.println();
  Serial.println(F("ESTADO DEL BRAZO"));
  Serial.println(F("--------------------------------"));

  Serial.print(F("Servos activados: "));
  Serial.println(servosActivados ? "SI" : "NO");

  Serial.print(F("Brazo ocupado: "));
  Serial.println(brazoOcupado ? "SI" : "NO");

  Serial.print(F("LED de inspeccion: "));
  Serial.println(nombreEstadoLedRgb());

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    Serial.print(i);
    Serial.print(F(" | "));
    Serial.print(servos[i].nombre);
    Serial.print(F(" | Canal "));
    Serial.print(servos[i].canal);
    Serial.print(F(" | Actual: "));
    Serial.print(posicionActual[i]);
    Serial.print(F(" us | Limites: "));
    Serial.print(servos[i].pulsoMin);
    Serial.print(F("-"));
    Serial.print(servos[i].pulsoMax);
    Serial.print(F(" | HOME: "));
    Serial.println(servos[i].pulsoHome);
  }

  Serial.println(F("--------------------------------"));
}

// =====================================================
// MOSTRAR UNA POSE
// =====================================================

void mostrarPose(
  const int pose[NUM_SERVOS],
  const char* nombre
) {
  Serial.println();

  Serial.print(F("POSE "));
  Serial.println(nombre);

  Serial.println(F("--------------------------------"));

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    Serial.print(servos[i].nombre);
    Serial.print(F(": "));
    Serial.print(pose[i]);
    Serial.println(F(" us"));
  }

  Serial.println(F("--------------------------------"));
}

// =====================================================
// MOSTRAR AYUDA
// =====================================================

void mostrarAyuda() {
  Serial.println(F("COMANDOS DISPONIBLES:"));
  Serial.println(F("  A          Activar servos en HOME"));
  Serial.println(F("  H          Ejecutar regreso seguro a HOME"));
  Serial.println(F("  R          Ejecutar secuencia de recogida"));
  Serial.println(F("  X          Desactivar todos los servos"));
  Serial.println(F("  ?          Mostrar estado actual"));
  Serial.println(F("  POSES      Mostrar poses guardadas"));
  Serial.println(F("  AYUDA      Mostrar esta ayuda"));
  Serial.println(F("  M0,1900    Mover servo 0 a 1900 us"));
  Serial.println(F("  M1,1700    Mover servo 1 a 1700 us"));
  Serial.println(F("  OK         Recoger y clasificar pieza OK"));
  Serial.println(F("  NOK        Recoger y clasificar pieza NOK"));
  Serial.println(F("  LED_OK     Encender indicador VERDE"));
  Serial.println(F("  LED_NOK    Encender indicador ROJO"));
  Serial.println(F("  LED_OFF    Apagar indicador"));
  Serial.println();
}

// =====================================================
// INDICADOR FISICO LED RGB
// =====================================================

void inicializarLedRgb() {
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);

  ledRgbOff();
}

// Aplica el estado logico de cada color.
// En catodo comun: HIGH = encendido.
// En anodo comun: LOW = encendido.
void aplicarLedRgb(bool rojoEncendido, bool verdeEncendido) {
  int nivelRojo;
  int nivelVerde;

  if (LED_RGB_ANODO_COMUN) {
    nivelRojo = rojoEncendido ? LOW : HIGH;
    nivelVerde = verdeEncendido ? LOW : HIGH;
  } else {
    nivelRojo = rojoEncendido ? HIGH : LOW;
    nivelVerde = verdeEncendido ? HIGH : LOW;
  }

  digitalWrite(PIN_LED_ROJO, nivelRojo);
  digitalWrite(PIN_LED_VERDE, nivelVerde);
}

void ledRgbOff() {
  aplicarLedRgb(false, false);
  estadoLedRgb = LED_ESTADO_OFF;
}

void ledRgbOk() {
  aplicarLedRgb(false, true);
  estadoLedRgb = LED_ESTADO_OK;
}

void ledRgbNok() {
  aplicarLedRgb(true, false);
  estadoLedRgb = LED_ESTADO_NOK;
}

bool procesarComandoLedRgb(const String& comando) {
  if (comando == "LED_OK") {
    ledRgbOk();
    Serial.println(F("LED_DONE"));
    return true;
  }

  if (comando == "LED_NOK") {
    ledRgbNok();
    Serial.println(F("LED_DONE"));
    return true;
  }

  if (comando == "LED_OFF") {
    ledRgbOff();
    Serial.println(F("LED_DONE"));
    return true;
  }

  return false;
}

const char* nombreEstadoLedRgb() {
  switch (estadoLedRgb) {
    case LED_ESTADO_OK:
      return "VERDE (OK)";

    case LED_ESTADO_NOK:
      return "ROJO (NOK)";

    case LED_ESTADO_OFF:
    default:
      return "APAGADO";
  }
}