import glob
import queue
import time
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


# Puedes escribir un puerto fijo, por ejemplo "/dev/ttyACM0".
# Con None se busca automáticamente el primer Arduino disponible.
PUERTO_ARDUINO: Optional[str] = None
BAUDIOS = 115200
TIEMPO_REINICIO_ARDUINO = 2.5
TIMEOUT_ACTIVACION = 10.0
TIMEOUT_MOVIMIENTO = 90.0
TIMEOUT_LED = 3.0


class ErrorBrazo(RuntimeError):
    """Error de conexión, protocolo o ejecución del brazo."""


def detectar_puerto_arduino() -> str:
    """Busca primero dispositivos Arduino/USB serial y devuelve su ruta."""
    if serial is None:
        raise ErrorBrazo(
            "PySerial no está instalado. Ejecuta: python3 -m pip install pyserial"
        )

    if list_ports is None:
        raise ErrorBrazo(
            "PySerial no está instalado. Ejecuta: python3 -m pip install pyserial"
        )

    puertos = list(list_ports.comports())

    def prioridad(puerto):
        dispositivo = (puerto.device or "").lower()
        descripcion = (puerto.description or "").lower()
        texto = f"{dispositivo} {descripcion}"

        if "arduino" in texto:
            return 0
        if "ttyacm" in texto:
            return 1
        if "ch340" in texto or "cp210" in texto or "usb serial" in texto:
            return 2
        if "ttyusb" in texto:
            return 3
        return 10

    puertos.sort(key=prioridad)

    for puerto in puertos:
        if prioridad(puerto) < 10:
            return puerto.device

    # Respaldo por si list_ports no muestra descripción suficiente.
    candidatos = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if candidatos:
        return candidatos[0]

    raise ErrorBrazo("No se encontró Arduino en /dev/ttyACM* ni /dev/ttyUSB*.")


class BrazoWorker(QThread):
    """
    Hilo persistente para comunicación serial con Arduino.

    La GUI deposita solicitudes en una cola. Todas las operaciones lentas
    (abrir puerto, esperar READY y esperar DONE) ocurren aquí, por lo que
    la ventana y el video no se congelan durante una trayectoria.
    """

    estado_cambiado = pyqtSignal(str)
    mensaje_recibido = pyqtSignal(str)
    conectado = pyqtSignal(str)
    desconectado = pyqtSignal()
    movimiento_iniciado = pyqtSignal(str)
    movimiento_terminado = pyqtSignal(str, float)
    led_cambiado = pyqtSignal(str)
    error_led = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, puerto: Optional[str] = PUERTO_ARDUINO):
        super().__init__()
        self.puerto_preferido = puerto
        self._serial = None
        self._cola = queue.Queue()
        self._detener = False
        self._estado = "DESCONECTADO"

    def solicitar_conexion(self, puerto: Optional[str] = None) -> None:
        self._cola.put(("CONECTAR", puerto))

    def solicitar_desconexion(self) -> None:
        self._cola.put(("DESCONECTAR", None))

    def solicitar_movimiento(self, comando: str) -> None:
        self._cola.put(("MOVER", comando.strip().upper()))

    def solicitar_led(self, estado: str) -> None:
        """Solicita el estado del LED RGB: OK, NOK u OFF."""
        estado_normalizado = estado.strip().upper()
        if estado_normalizado not in {"OK", "NOK", "OFF"}:
            raise ValueError(f"Estado LED no permitido: {estado}")
        self._cola.put(("LED", estado_normalizado))

    def detener(self) -> None:
        self._detener = True
        self._cola.put(("DETENER", None))
        self.wait(3000)

    def run(self) -> None:
        while not self._detener:
            try:
                accion, valor = self._cola.get(timeout=0.1)
            except queue.Empty:
                continue

            if accion == "DETENER":
                break

            try:
                if accion == "CONECTAR":
                    self._conectar(valor)
                elif accion == "DESCONECTAR":
                    self._desconectar()
                elif accion == "MOVER":
                    self._ejecutar_movimiento(valor)
                elif accion == "LED":
                    # Un fallo del indicador no debe tumbar la conexión del brazo
                    # ni registrarse como fallo mecánico de clasificación.
                    try:
                        self._ejecutar_led(valor)
                    except Exception as exc:
                        self.error_led.emit(str(exc))
            except Exception as exc:
                self._cerrar_puerto()
                self._cambiar_estado("ERROR")
                self.error.emit(str(exc))

        self._cerrar_puerto()

    def _cambiar_estado(self, estado: str) -> None:
        self._estado = estado
        self.estado_cambiado.emit(estado)

    def _conectar(self, puerto_solicitado: Optional[str]) -> None:
        if self._serial is not None and self._serial.is_open:
            self.mensaje_recibido.emit("El brazo ya está conectado.")
            return

        if serial is None:
            raise ErrorBrazo(
                "PySerial no está instalado. Ejecuta: python3 -m pip install pyserial"
            )

        puerto = puerto_solicitado or self.puerto_preferido or detectar_puerto_arduino()
        self._cambiar_estado("CONECTANDO")
        self.mensaje_recibido.emit(f"Abriendo puerto {puerto}...")

        try:
            self._serial = serial.Serial(
                port=puerto,
                baudrate=BAUDIOS,
                timeout=0.2,
                write_timeout=2.0,
            )
        except (OSError, ValueError) as exc:
            raise ErrorBrazo(f"No se pudo abrir {puerto}: {exc}") from exc

        # Muchos Arduino se reinician al abrir el puerto.
        time.sleep(TIEMPO_REINICIO_ARDUINO)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

        # El programa del Arduino supone que físicamente está en HOME.
        self._enviar_linea("A")
        self._esperar_respuesta({"READY"}, TIMEOUT_ACTIVACION)

        self._cambiar_estado("LISTO")
        self.conectado.emit(puerto)

    def _desconectar(self) -> None:
        self._cerrar_puerto()
        self._cambiar_estado("DESCONECTADO")
        self.desconectado.emit()

    def _ejecutar_movimiento(self, comando: str) -> None:
        if comando not in {"OK", "NOK", "H"}:
            raise ErrorBrazo(f"Comando de brazo no permitido: {comando}")

        if self._serial is None or not self._serial.is_open:
            raise ErrorBrazo("El brazo no está conectado.")

        if self._estado == "MOVIENDO":
            raise ErrorBrazo("El brazo ya está ejecutando una trayectoria.")

        self._cambiar_estado("MOVIENDO")
        self.movimiento_iniciado.emit(comando)

        # El tiempo del brazo se mide exactamente desde antes de enviar el
        # comando serial hasta recibir DONE del Arduino.
        inicio_movimiento = time.monotonic()
        self._enviar_linea(comando)
        self._esperar_respuesta({"DONE"}, TIMEOUT_MOVIMIENTO)
        tiempo_movimiento_s = time.monotonic() - inicio_movimiento

        self._cambiar_estado("LISTO")
        self.movimiento_terminado.emit(comando, tiempo_movimiento_s)

    def _ejecutar_led(self, estado: str) -> None:
        """Actualiza el LED físico sin cambiar el estado MOVIENDO/LISTO del brazo."""
        estado = estado.strip().upper()
        comandos = {
            "OK": "LED_OK",
            "NOK": "LED_NOK",
            "OFF": "LED_OFF",
        }

        if estado not in comandos:
            raise ErrorBrazo(f"Estado LED no permitido: {estado}")

        if self._serial is None or not self._serial.is_open:
            raise ErrorBrazo("No se puede actualizar el LED: brazo desconectado.")

        self._enviar_linea(comandos[estado])
        self._esperar_respuesta({"LED_DONE"}, TIMEOUT_LED)
        self.led_cambiado.emit(estado)

    def _enviar_linea(self, comando: str) -> None:
        if self._serial is None or not self._serial.is_open:
            raise ErrorBrazo("El puerto serial no está abierto.")

        try:
            self._serial.write(f"{comando}\n".encode("utf-8"))
            self._serial.flush()
        except (OSError, ValueError) as exc:
            raise ErrorBrazo(f"No se pudo enviar '{comando}': {exc}") from exc

        self.mensaje_recibido.emit(f"Python -> {comando}")

    def _leer_linea(self) -> Optional[str]:
        if self._serial is None or not self._serial.is_open:
            raise ErrorBrazo("El puerto serial no está abierto.")

        try:
            datos = self._serial.readline()
        except (OSError, ValueError) as exc:
            raise ErrorBrazo(f"Error leyendo Arduino: {exc}") from exc

        if not datos:
            return None

        linea = datos.decode("utf-8", errors="replace").strip()
        if linea:
            self.mensaje_recibido.emit(f"Arduino -> {linea}")
            return linea
        return None

    def _esperar_respuesta(self, finales: set[str], timeout: float) -> str:
        inicio = time.monotonic()

        while time.monotonic() - inicio < timeout:
            if self._detener:
                raise ErrorBrazo("Comunicación detenida por cierre de la aplicación.")

            linea = self._leer_linea()
            if linea is None:
                continue

            if linea.startswith("ERROR"):
                raise ErrorBrazo(linea)

            if linea in finales:
                return linea

        esperadas = ", ".join(sorted(finales))
        raise ErrorBrazo(f"Tiempo agotado esperando respuesta: {esperadas}")

    def _cerrar_puerto(self) -> None:
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
        self._serial = None