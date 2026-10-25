import sys
import os
import time
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSlider,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QSizePolicy,
    QScrollArea,
    QFrame,
)

from inspeccion_core_f import (
    MotorInspeccion,
    USAR_SHAPE_RANURAS,
    AGUJEROS_ESPERADOS,
    RANURAS_ESPERADAS,
    NUM_MUESTRAS_MASTER,
    buscar_resultado_por_id,
)
from controlador_brazo import BrazoWorker


# ==============================
# FUNCIONES DE APOYO
# ==============================

def cv_bgr_a_pixmap(imagen_bgr, ancho_objetivo=None, alto_objetivo=None):
    """Convierte una imagen OpenCV BGR a QPixmap para mostrarla en QLabel."""
    if imagen_bgr is None:
        return QPixmap()

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = imagen_rgb.shape
    bytes_por_linea = ch * w

    qimg = QImage(
        imagen_rgb.data,
        w,
        h,
        bytes_por_linea,
        QImage.Format.Format_RGB888
    ).copy()

    pixmap = QPixmap.fromImage(qimg)

    if ancho_objetivo is not None and alto_objetivo is not None:
        pixmap = pixmap.scaled(
            ancho_objetivo,
            alto_objetivo,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    return pixmap


# Colores por estado, equivalentes en CSS a los BGR que usa
# color_por_estado() en inspeccion_core.py (para que el criterio de
# color sea el mismo en la ventana de video y en este panel).
def color_estado_css(estado):
    if estado == "OK":
        return "#2ecc71"

    if estado in ("SIN_MASTER", "SIN_NOMINAL", "EVALUANDO...", "SIN_PIEZA"):
        return "#f1c40f"

    if estado is None:
        return "#95a5a6"

    return "#e74c3c"


COLOR_TITULO = "#00e5ff"
COLOR_TEXTO_NORMAL = "#ecf0f1"
COLOR_TEXTO_APAGADO = "#7f8c8d"


# ==============================
# PANEL DE RESULTADOS (WIDGETS NATIVOS)
# ==============================

class PanelResultadosWidget(QWidget):
    """
    Panel de inspección con widgets nativos de Qt (un QLabel por línea)
    en vez de una imagen OpenCV escalada.

    Por qué: el panel anterior era una imagen de tamaño fijo (950x1100)
    dibujada con cv2.putText, después escalada hacia abajo para caber
    en un QLabel chico — al escalar una imagen con texto, el texto se
    ve borroso, y agrandar el QLabel solo agranda esa imagen borrosa,
    no la nitidez del texto.

    Con QLabel nativos, el texto se renderiza nítido a cualquier
    tamaño (Qt usa el motor de fuentes del sistema), y como todo el
    panel vive dentro de un QScrollArea (ver VentanaInspeccion), nunca
    se corta información aunque la ventana sea chica: simplemente
    aparece una barra de desplazamiento.

    Los QLabel se crean UNA sola vez en __init__ y se reutilizan en
    cada frame vía actualizar() — no se reconstruye el layout 30+
    veces por segundo, solo se cambia el texto/color de cada label.
    """

    def __init__(self):
        super().__init__()

        self._labels = {}

        layout = QVBoxLayout()
        layout.setSpacing(3)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        def titulo(texto):
            lbl = QLabel(texto)
            lbl.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {COLOR_TITULO}; "
                "margin-top: 10px; border-bottom: 1px solid #444;"
            )
            layout.addWidget(lbl)

        def campo(nombre, texto_inicial="--"):
            lbl = QLabel(texto_inicial)
            lbl.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXTO_NORMAL};")
            lbl.setWordWrap(True)
            layout.addWidget(lbl)
            self._labels[nombre] = lbl

        # --- Encabezado ---
        titulo("PANEL DE INSPECCION")
        campo("resultado_frame")
        campo("veredicto_estable")
        campo("shape_master")
        campo("captura_master")

        # --- Contorno exterior ---
        titulo("CONTORNO EXTERIOR")
        campo("p1")
        campo("area")
        campo("relleno_rect")
        campo("esquinas")
        campo("vertices")
        campo("centroide")
        campo("orientacion")

        # --- Shape Matching ---
        titulo("SHAPE MATCHING")
        campo("shape_mask")
        campo("shape_exterior")
        if USAR_SHAPE_RANURAS:
            campo("shape_r1")
            campo("shape_r3")
        campo("shape_global")

        # --- Posición de features ---
        titulo("POSICION DE FEATURES")
        campo("posicion_estado")
        for esperado in AGUJEROS_ESPERADOS:
            campo(f"pos_{esperado['id']}")
        for esperado in RANURAS_ESPERADAS:
            campo(f"pos_{esperado['id']}")

        # --- Agujeros ---
        titulo("AGUJEROS")
        campo("agujeros_titulo")
        for esperado in AGUJEROS_ESPERADOS:
            campo(f"agujero_{esperado['id']}")

        # --- Ranuras ---
        titulo("RANURAS")
        campo("ranuras_titulo")
        for esperado in RANURAS_ESPERADAS:
            campo(f"ranura_{esperado['id']}")

        self.setLayout(layout)
        self.setStyleSheet("background-color: #1c1c1c;")

    def _set(self, nombre, texto, color=COLOR_TEXTO_NORMAL, negrita=False):
        lbl = self._labels.get(nombre)
        if lbl is None:
            return
        lbl.setText(texto)
        peso = "font-weight: bold;" if negrita else ""
        lbl.setStyleSheet(f"font-size: 12px; color: {color}; {peso}")

    def actualizar(self, data):
        resultado_global = data.get("resultado_global")
        veredicto_estable = data.get("veredicto_estable")
        resultado_exterior = data.get("resultado_exterior")
        resultado_shape = data.get("resultado_shape")
        resultado_posicion = data.get("resultado_posicion")
        resultados_agujeros = data.get("resultados_agujeros") or []
        resultados_ranuras = data.get("resultados_ranuras") or []
        agujeros = data.get("agujeros") or []
        ranuras_detectadas = data.get("ranuras_detectadas") or []

        # Si el objeto no pertenece a la familia de la placa, se limpia el
        # panel para no mostrar medidas o defectos de un objeto ajeno.
        if resultado_global == "SIN_PIEZA":
            for nombre in self._labels:
                self._set(nombre, "--", COLOR_TEXTO_APAGADO)

            self._set(
                "resultado_frame",
                "PIEZA NO DETECTADA",
                "#f1c40f",
                negrita=True
            )
            self._set(
                "veredicto_estable",
                "PIEZA NO DETECTADA",
                "#f1c40f",
                negrita=True
            )
            return

        # --- Encabezado ---
        self._set(
            "resultado_frame",
            f"Resultado frame: {resultado_global}",
            color_estado_css(resultado_global),
            negrita=True
        )
        self._set(
            "veredicto_estable",
            f"Veredicto estable: {veredicto_estable}",
            color_estado_css(veredicto_estable),
            negrita=True
        )

        if data.get("shape_master_cargada"):
            n = data.get("shape_master_num_muestras", "?")
            self._set("shape_master", f"Shape master: CARGADA ({n} muestras)", "#2ecc71")
        else:
            self._set("shape_master", "Shape master: NO CARGADA", "#f1c40f")

        capturas = data.get("capturas_master_en_curso", 0)
        if capturas:
            self._set(
                "captura_master",
                f"Captura master en curso: {capturas}/{NUM_MUESTRAS_MASTER}",
                "#f1c40f"
            )
        else:
            self._set("captura_master", "", COLOR_TEXTO_APAGADO)

        # --- Contorno exterior ---
        if resultado_exterior is None:
            self._set("p1", "P1: No detectado", "#e74c3c")
            for campo in ("area", "relleno_rect", "esquinas", "vertices", "centroide", "orientacion"):
                self._set(campo, "--", COLOR_TEXTO_APAGADO)
        else:
            self._set(
                "p1",
                f"P1: {resultado_exterior['lado_mayor_mm']:.2f} x {resultado_exterior['lado_menor_mm']:.2f} mm | {resultado_exterior['estado_contorno']}",
                color_estado_css(resultado_exterior["estado_contorno"])
            )
            self._set(
                "area",
                f"Area: {resultado_exterior['area_contorno_mm2']:.1f} mm2 | Error: {resultado_exterior['error_area_porc']:.2f}%",
                color_estado_css(resultado_exterior["estado_contorno"])
            )
            self._set(
                "relleno_rect",
                f"Relleno rect: {resultado_exterior['factor_relleno_rect']:.4f} | {resultado_exterior['estado_forma_exterior']}",
                color_estado_css(resultado_exterior["estado_forma_exterior"])
            )
            self._set(
                "esquinas",
                f"Esquinas: {resultado_exterior['relleno_min_esquina']:.3f} | {resultado_exterior['estado_esquinas']}",
                color_estado_css(resultado_exterior["estado_esquinas"])
            )
            self._set(
                "vertices",
                f"Vertices: {resultado_exterior['vertices_exterior']} | {resultado_exterior['estado_vertices_exterior']}",
                color_estado_css(resultado_exterior["estado_vertices_exterior"])
            )
            self._set(
                "centroide",
                f"Centroide: X={resultado_exterior['cx_mm']:.2f}, Y={resultado_exterior['cy_mm']:.2f} mm | {resultado_exterior['estado_centroide']}",
                color_estado_css(resultado_exterior["estado_centroide"])
            )
            self._set(
                "orientacion",
                f"Orientacion: {resultado_exterior['angulo']:.2f} grados | {resultado_exterior['estado_orientacion']}",
                color_estado_css(resultado_exterior["estado_orientacion"])
            )

        # --- Shape Matching ---
        if resultado_shape is None:
            self._set("shape_mask", "Shape: sin evaluacion", COLOR_TEXTO_APAGADO)
            self._set("shape_exterior", "", COLOR_TEXTO_APAGADO)
            self._set("shape_global", "", COLOR_TEXTO_APAGADO)
        else:
            if resultado_shape["score_mascara"] is None:
                self._set("shape_mask", "Shape mask: SIN MASTER", color_estado_css(resultado_shape["estado_mascara"]))
            else:
                self._set(
                    "shape_mask",
                    f"Shape mask: {resultado_shape['score_mascara']:.5f} | {resultado_shape['estado_mascara']}",
                    color_estado_css(resultado_shape["estado_mascara"])
                )

            if resultado_shape["score_exterior"] is None:
                self._set("shape_exterior", "Shape exterior: SIN MASTER", color_estado_css(resultado_shape["estado_exterior"]))
            else:
                self._set(
                    "shape_exterior",
                    f"Shape exterior: {resultado_shape['score_exterior']:.5f} | {resultado_shape['estado_exterior']}",
                    color_estado_css(resultado_shape["estado_exterior"])
                )

            if USAR_SHAPE_RANURAS:
                if resultado_shape["score_r1"] is None:
                    self._set("shape_r1", "Shape R1: SIN REF", color_estado_css(resultado_shape["estado_r1"]))
                else:
                    self._set(
                        "shape_r1",
                        f"Shape R1: {resultado_shape['score_r1']:.5f} | {resultado_shape['estado_r1']}",
                        color_estado_css(resultado_shape["estado_r1"])
                    )

                if resultado_shape["score_r3"] is None:
                    self._set("shape_r3", "Shape R3: SIN REF", color_estado_css(resultado_shape["estado_r3"]))
                else:
                    self._set(
                        "shape_r3",
                        f"Shape R3: {resultado_shape['score_r3']:.5f} | {resultado_shape['estado_r3']}",
                        color_estado_css(resultado_shape["estado_r3"])
                    )

            self._set(
                "shape_global",
                f"Shape global: {resultado_shape['estado_shape_global']}",
                color_estado_css(resultado_shape["estado_shape_global"]),
                negrita=True
            )

        # --- Posición de features ---
        if resultado_posicion is None:
            self._set("posicion_estado", "Posicion: sin evaluacion", COLOR_TEXTO_APAGADO)
            ids_posicion = [e["id"] for e in AGUJEROS_ESPERADOS] + [e["id"] for e in RANURAS_ESPERADAS]
            for id_feat in ids_posicion:
                self._set(f"pos_{id_feat}", "--", COLOR_TEXTO_APAGADO)
        else:
            self._set(
                "posicion_estado",
                f"Estado posicion: {resultado_posicion['estado_posicion_global']}",
                color_estado_css(resultado_posicion["estado_posicion_global"]),
                negrita=True
            )

            ids_posicion = [e["id"] for e in AGUJEROS_ESPERADOS] + [e["id"] for e in RANURAS_ESPERADAS]
            for id_feat in ids_posicion:
                r = buscar_resultado_por_id(resultado_posicion["resultados"], id_feat)
                nombre_campo = f"pos_{id_feat}"

                if r is None:
                    self._set(nombre_campo, f"{id_feat}: --", COLOR_TEXTO_APAGADO)
                elif r["estado"] == "SIN_NOMINAL":
                    self._set(nombre_campo, f"{id_feat}: sin posicion nominal", "#f1c40f")
                elif r["medido"] is None:
                    self._set(nombre_campo, f"{id_feat}: AUSENTE | {r['estado']}", color_estado_css(r["estado"]))
                else:
                    self._set(
                        nombre_campo,
                        f"{id_feat}: error {r['error_mm']:.2f} mm "
                        f"(X={r['medido'][0]:.1f}, Y={r['medido'][1]:.1f}) | {r['estado']}",
                        color_estado_css(r["estado"])
                    )

        # --- Agujeros ---
        self._set("agujeros_titulo", f"Agujeros detectados: {len(agujeros)}", COLOR_TEXTO_NORMAL, negrita=True)

        for esperado in AGUJEROS_ESPERADOS:
            id_ag = esperado["id"]
            r = buscar_resultado_por_id(resultados_agujeros, id_ag)
            nombre_campo = f"agujero_{id_ag}"

            if r is None:
                self._set(nombre_campo, f"{id_ag}: --", COLOR_TEXTO_APAGADO)
            elif r["medido"] is None:
                self._set(
                    nombre_campo,
                    f"{id_ag}: AUSENTE | Nom {r['nominal']:.1f} mm | {r['estado']}",
                    color_estado_css(r["estado"])
                )
            else:
                self._set(
                    nombre_campo,
                    f"{id_ag}: {r['medido']:.2f} mm | Lim {r['lim_inf']:.1f}-{r['lim_sup']:.1f} | {r['estado']}",
                    color_estado_css(r["estado"])
                )

        # --- Ranuras ---
        self._set("ranuras_titulo", f"Ranuras detectadas: {len(ranuras_detectadas)}", COLOR_TEXTO_NORMAL, negrita=True)

        for esperado in RANURAS_ESPERADAS:
            id_r = esperado["id"]
            r = buscar_resultado_por_id(resultados_ranuras, id_r)
            nombre_campo = f"ranura_{id_r}"

            if r is None:
                self._set(nombre_campo, f"{id_r}: --", COLOR_TEXTO_APAGADO)
            elif r["cantidad_detectada"] == 0:
                self._set(
                    nombre_campo,
                    f"{id_r}: AUSENTE (0/{r['cantidad_esperada']}) | NOK",
                    color_estado_css("NOK")
                )
            elif r["cantidad_detectada"] != r["cantidad_esperada"]:
                self._set(
                    nombre_campo,
                    f"{id_r}: EXCESO ({r['cantidad_detectada']}/{r['cantidad_esperada']}) | NOK",
                    color_estado_css("NOK")
                )
            else:
                self._set(
                    nombre_campo,
                    f"{id_r}: {r['largo_medido']:.2f} x {r['ancho_medido']:.2f} mm | {r['estado']}",
                    color_estado_css(r["estado"])
                )


# ==============================
# HILO DE CÁMARA
# ==============================

class CamaraWorker(QThread):
    frame_actualizado = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, motor):
        super().__init__()
        self.motor = motor
        self._activo = True

    def run(self):
        while self._activo:
            try:
                data = self.motor.leer_y_procesar()
                self.frame_actualizado.emit(data)
            except Exception as e:
                self.error.emit(str(e))
                self.msleep(300)

            self.msleep(30)

    def detener(self):
        self._activo = False
        self.wait(1500)


# ==============================
# VENTANA PRINCIPAL
# ==============================

class VentanaInspeccion(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sistema de Inspección Automatizada")
        self.resize(1500, 900)

        self.motor = MotorInspeccion()
        self.ultimo_data = None

        self.estado_brazo = "DESCONECTADO"
        self.brazo_conectado = False
        self.brazo_ocupado = False

        # Estado del indicador físico. Se guarda el último solicitado para
        # no saturar el puerto serial enviando el mismo comando cada frame.
        self.estado_led_deseado = "OFF"
        self.ultimo_estado_led_solicitado = None

        # Datos de la inspección que está esperando a que el brazo termine.
        self.id_inspeccion_pendiente = None
        self.inicio_ciclo_pendiente = None
        self.tiempo_inspeccion_pendiente_s = None

        self.crear_interfaz()
        self.configurar_atajos_teclado()
        self.cargar_historial()

        self.worker = CamaraWorker(self.motor)
        self.worker.frame_actualizado.connect(self.actualizar_frame)
        self.worker.error.connect(self.mostrar_error)
        self.worker.start()

        self.worker_brazo = BrazoWorker()
        self.worker_brazo.estado_cambiado.connect(self.actualizar_estado_brazo)
        self.worker_brazo.mensaje_recibido.connect(self.mostrar_mensaje_brazo)
        self.worker_brazo.conectado.connect(self.brazo_conectado_correctamente)
        self.worker_brazo.desconectado.connect(self.brazo_desconectado_correctamente)
        self.worker_brazo.movimiento_iniciado.connect(self.movimiento_brazo_iniciado)
        self.worker_brazo.movimiento_terminado.connect(self.movimiento_brazo_terminado)
        self.worker_brazo.error.connect(self.mostrar_error_brazo)
        self.worker_brazo.start()

        self.actualizar_disponibilidad_brazo()

    # ------------------------------
    # Construcción de interfaz
    # ------------------------------

    def crear_interfaz(self):
        widget_central = QWidget()
        layout_principal = QVBoxLayout()

        layout_superior = QHBoxLayout()

        # Imagen de inspección
        grupo_imagen = QGroupBox("Imagen en vivo")
        layout_imagen = QVBoxLayout()
        self.label_imagen = QLabel("Esperando cámara...")
        self.label_imagen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_imagen.setMinimumSize(640, 380)
        self.label_imagen.setStyleSheet("background-color: #111; color: white;")
        self.label_imagen.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout_imagen.addWidget(self.label_imagen)
        grupo_imagen.setLayout(layout_imagen)
        layout_superior.addWidget(grupo_imagen, 3)

        # Panel y estado
        grupo_estado = QGroupBox("Estado de inspección")
        layout_estado = QVBoxLayout()

        self.label_resultado = QLabel("Resultado: --")
        self.label_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_resultado.setStyleSheet("font-size: 30px; font-weight: bold; padding: 10px;")

        self.label_registro = QLabel("Registro: --")
        self.label_registro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_registro.setStyleSheet("font-size: 16px; padding: 6px;")

        self.label_mensaje = QLabel("Listo.")
        self.label_mensaje.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_mensaje.setWordWrap(True)
        self.label_mensaje.setStyleSheet("font-size: 14px; color: #444; padding: 6px;")

        self.label_tiempos = QLabel(
            "Tiempos | Inspección: -- s | Brazo: -- s | Clasificación: -- s | Ciclo: -- s"
        )
        self.label_tiempos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_tiempos.setWordWrap(True)
        self.label_tiempos.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #2c3e50; padding: 6px; "
            "background-color: #ecf0f1; border-radius: 4px;"
        )

        self.label_estado_brazo = QLabel("Brazo: DESCONECTADO")
        self.label_estado_brazo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_estado_brazo.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 6px; "
            "background-color: #bdc3c7;"
        )

        self.label_puerto_brazo = QLabel("Puerto del brazo: detección automática")
        self.label_puerto_brazo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_puerto_brazo.setStyleSheet("font-size: 12px; color: #555;")

        layout_estado.addWidget(self.label_resultado)
        layout_estado.addWidget(self.label_registro)
        layout_estado.addWidget(self.label_mensaje)
        layout_estado.addWidget(self.label_tiempos)
        layout_estado.addWidget(self.label_estado_brazo)
        layout_estado.addWidget(self.label_puerto_brazo)
 
        # --- Panel de resultados: widgets nativos dentro de un
        # QScrollArea, en vez de una imagen OpenCV escalada. Ver
        # PanelResultadosWidget para la explicación completa. El
        # QScrollArea asegura que, sin importar cuán chico esté el
        # panel, siempre puedes desplazarte para leer todo con texto
        # nítido — nunca se ve borroso ni se corta información.
        self.panel_widget = PanelResultadosWidget()

        self.scroll_panel = QScrollArea()
        self.scroll_panel.setWidgetResizable(True)
        self.scroll_panel.setWidget(self.panel_widget)
        self.scroll_panel.setMinimumSize(380, 420)
        self.scroll_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout_estado.addWidget(self.scroll_panel)

        grupo_estado.setLayout(layout_estado)
        layout_superior.addWidget(grupo_estado, 2)

        layout_principal.addLayout(layout_superior, 4)

        # Botón para mostrar/ocultar controles.
        # Al ocultar este panel, la tabla de historial gana más espacio vertical.
        self.btn_toggle_controles = QPushButton("Ocultar controles ▲")
        self.btn_toggle_controles.clicked.connect(self.toggle_panel_controles)
        layout_principal.addWidget(self.btn_toggle_controles)

        # Botones y ajustes retraíbles
        self.grupo_controles = QGroupBox("Controles de inspección")
        layout_botones = QGridLayout()
        layout_botones.setContentsMargins(10, 10, 10, 10)
        layout_botones.setHorizontalSpacing(10)
        layout_botones.setVerticalSpacing(8)

        self.btn_registrar = QPushButton("Registrar inspección (i)")
        self.btn_registrar.clicked.connect(self.registrar_inspeccion)

        self.btn_pausar = QPushButton("Pausar registro")
        self.btn_pausar.clicked.connect(self.toggle_registro)

        self.btn_master = QPushButton("Capturar master")
        self.btn_master.clicked.connect(self.capturar_master)

        self.btn_reiniciar_master = QPushButton("Reiniciar master pendiente")
        self.btn_reiniciar_master.clicked.connect(self.reiniciar_master)

        self.btn_refrescar = QPushButton("Refrescar historial")
        self.btn_refrescar.clicked.connect(self.cargar_historial)

        self.btn_conectar_brazo = QPushButton("Conectar y activar brazo")
        self.btn_conectar_brazo.clicked.connect(self.conectar_desconectar_brazo)

        self.btn_home_brazo = QPushButton("Enviar brazo a HOME")
        self.btn_home_brazo.clicked.connect(self.enviar_home_brazo)
        self.btn_home_brazo.setEnabled(False)

        layout_botones.addWidget(self.btn_registrar, 0, 0)
        layout_botones.addWidget(self.btn_pausar, 0, 1)
        layout_botones.addWidget(self.btn_master, 0, 2)
        layout_botones.addWidget(self.btn_reiniciar_master, 0, 3)
        layout_botones.addWidget(self.btn_refrescar, 0, 4)
        layout_botones.addWidget(self.btn_conectar_brazo, 0, 5)
        layout_botones.addWidget(self.btn_home_brazo, 0, 6)

        # Parámetros simples de segmentación
        self.slider_umbral = QSlider(Qt.Orientation.Horizontal)
        self.slider_umbral.setRange(0, 255)
        self.slider_umbral.setValue(120)
        self.slider_umbral.valueChanged.connect(self.actualizar_parametros)
        self.label_umbral = QLabel("Umbral: 120")

        self.check_invertir = QCheckBox("Invertir binaria")
        self.check_invertir.stateChanged.connect(self.actualizar_parametros)

        self.slider_area_int = QSlider(Qt.Orientation.Horizontal)
        self.slider_area_int.setRange(0, 20000)
        self.slider_area_int.setValue(400)
        self.slider_area_int.valueChanged.connect(self.actualizar_parametros)
        self.label_area_int = QLabel("Área interna min: 400")

        self.slider_circularidad = QSlider(Qt.Orientation.Horizontal)
        self.slider_circularidad.setRange(0, 100)
        self.slider_circularidad.setValue(85)
        self.slider_circularidad.valueChanged.connect(self.actualizar_parametros)
        self.label_circularidad = QLabel("Circularidad: 0.85")

        self.slider_area_ranura = QSlider(Qt.Orientation.Horizontal)
        self.slider_area_ranura.setRange(0, 10000)
        self.slider_area_ranura.setValue(200)
        self.slider_area_ranura.valueChanged.connect(self.actualizar_parametros)
        self.label_area_ranura = QLabel("Área ranura min: 200")

        layout_botones.addWidget(self.label_umbral, 1, 0)
        layout_botones.addWidget(self.slider_umbral, 1, 1)
        layout_botones.addWidget(self.check_invertir, 1, 2)

        layout_botones.addWidget(self.label_area_int, 2, 0)
        layout_botones.addWidget(self.slider_area_int, 2, 1)
        layout_botones.addWidget(self.label_circularidad, 2, 2)
        layout_botones.addWidget(self.slider_circularidad, 2, 3)

        layout_botones.addWidget(self.label_area_ranura, 3, 0)
        layout_botones.addWidget(self.slider_area_ranura, 3, 1)

        self.grupo_controles.setLayout(layout_botones)
        layout_principal.addWidget(self.grupo_controles)

        # Indicadores visibles de operación. Se calculan desde SQLite y
        # representan toda la sesión/base, no solo las últimas 50 filas.
        grupo_indicadores = QGroupBox("Indicadores de operación")
        layout_indicadores = QGridLayout()
        layout_indicadores.setContentsMargins(10, 10, 10, 10)
        layout_indicadores.setHorizontalSpacing(10)
        layout_indicadores.setVerticalSpacing(8)

        self.label_ind_total = QLabel("Inspecciones totales\n0")
        self.label_ind_ok = QLabel("OK\n0")
        self.label_ind_nok = QLabel("NOK\n0")
        self.label_ind_tiempo = QLabel("Tiempo promedio del ciclo\n-- s")

        tarjetas_indicadores = [
            self.label_ind_total,
            self.label_ind_ok,
            self.label_ind_nok,
            self.label_ind_tiempo,
        ]

        for tarjeta in tarjetas_indicadores:
            tarjeta.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tarjeta.setMinimumHeight(72)
            tarjeta.setWordWrap(True)
            tarjeta.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 8px; "
                "background-color: #ecf0f1; color: #2c3e50; "
                "border: 1px solid #bdc3c7; border-radius: 6px;"
            )

        # Colores semánticos suaves para lectura rápida.
        self.label_ind_ok.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 8px; "
            "background-color: #d5f5e3; color: #196f3d; "
            "border: 1px solid #82e0aa; border-radius: 6px;"
        )
        self.label_ind_nok.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 8px; "
            "background-color: #fadbd8; color: #922b21; "
            "border: 1px solid #f1948a; border-radius: 6px;"
        )

        layout_indicadores.addWidget(self.label_ind_total, 0, 0)
        layout_indicadores.addWidget(self.label_ind_ok, 0, 1)
        layout_indicadores.addWidget(self.label_ind_nok, 0, 2)
        layout_indicadores.addWidget(self.label_ind_tiempo, 0, 3)

        grupo_indicadores.setLayout(layout_indicadores)
        layout_principal.addWidget(grupo_indicadores)

        # Historial
        grupo_historial = QGroupBox("Historial SQLite")
        layout_historial = QVBoxLayout()

        self.tabla = QTableWidget()
        self.tabla.setColumnCount(10)
        self.tabla.setHorizontalHeaderLabels([
            "ID", "timestamp", "Resultado", "Causa", "Contorno", "C1", "C2", "C3", "R1", "R3"
        ])
        header_tabla = self.tabla.horizontalHeader()
        if header_tabla is not None:
            header_tabla.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout_historial.addWidget(self.tabla)
        grupo_historial.setLayout(layout_historial)
        layout_principal.addWidget(grupo_historial, 3)

        widget_central.setLayout(layout_principal)

        # --- Scroll general de TODA la interfaz ---
        # Antes solo el panel de resultados tenía su propio scroll.
        # Envolver también el widget_central completo en un QScrollArea
        # asegura que, si la ventana es más chica que el contenido
        # (imagen + panel + controles + tabla), aparece una barra de
        # desplazamiento vertical para toda la interfaz en vez de que
        # los controles queden apretados o cortados. setWidgetResizable
        # es lo que permite que el ancho interno siga ajustándose al
        # viewport (los layouts "Expanding" adentro se comportan igual
        # que antes) mientras que la altura sí puede exceder el
        # viewport y activar el scroll.
        self.scroll_general = QScrollArea()
        self.scroll_general.setWidgetResizable(True)
        self.scroll_general.setWidget(widget_central)
        self.scroll_general.setFrameShape(QFrame.Shape.NoFrame)

        self.setCentralWidget(self.scroll_general)

    # ------------------------------
    # Panel retraíble de controles
    # ------------------------------

    def toggle_panel_controles(self):
        """
        Muestra u oculta el panel de botones y ajustes.
        Cuando se oculta, la tabla de registros tiene más espacio visible.
        """
        controles_visibles = self.grupo_controles.isVisible()

        if controles_visibles:
            self.grupo_controles.hide()
            self.btn_toggle_controles.setText("Mostrar controles ▼")
        else:
            self.grupo_controles.show()
            self.btn_toggle_controles.setText("Ocultar controles ▲")
        
    # ------------------------------
    # Atajos de teclado
    # ------------------------------

    def configurar_atajos_teclado(self):
        """
        Permite registrar con la tecla i, además del botón.
        QShortcut funciona aunque el foco esté en un botón o control de la ventana.
        """
        self.shortcut_registrar = QShortcut(QKeySequence("I"), self)
        self.shortcut_registrar.activated.connect(self.registrar_inspeccion)

    # ------------------------------
    # Eventos y actualizaciones
    # ------------------------------

    def actualizar_parametros(self):
        umbral = self.slider_umbral.value()
        invertir = 1 if self.check_invertir.isChecked() else 0
        area_int = self.slider_area_int.value()
        circularidad = self.slider_circularidad.value() / 100.0
        area_ranura = self.slider_area_ranura.value()

        self.label_umbral.setText(f"Umbral: {umbral}")
        self.label_area_int.setText(f"Área interna min: {area_int}")
        self.label_circularidad.setText(f"Circularidad: {circularidad:.2f}")
        self.label_area_ranura.setText(f"Área ranura min: {area_ranura}")

        self.motor.set_parametros(
            umbral=umbral,
            invertir=invertir,
            area_interna_min=area_int,
            circularidad_min=circularidad,
            area_ranura_min=area_ranura,
        )

    def actualizar_frame(self, data):
        self.ultimo_data = data

        pixmap_salida = cv_bgr_a_pixmap(
            data.get("salida"),
            self.label_imagen.width(),
            self.label_imagen.height()
        )
        self.label_imagen.setPixmap(pixmap_salida)

        # El panel ya no se dibuja como imagen: se actualizan los
        # QLabel nativos del panel con los mismos datos (resultado_*,
        # resultados_agujeros, etc.) que antes se usaban para dibujar
        # texto sobre una imagen OpenCV.
        self.panel_widget.actualizar(data)

        resultado = data.get("resultado_global", "--")
        veredicto = data.get("veredicto_estable", "--")
        texto_registro = data.get("texto_registro", "--")

        resultado_ui = (
            "PIEZA NO DETECTADA"
            if resultado == "SIN_PIEZA"
            else resultado
        )
        veredicto_ui = (
            "PIEZA NO DETECTADA"
            if resultado == "SIN_PIEZA" or veredicto == "SIN_PIEZA"
            else veredicto
        )

        self.label_resultado.setText(f"Resultado: {resultado_ui}")
        self.label_registro.setText(texto_registro)

        if resultado == "OK":
            self.label_resultado.setStyleSheet("font-size: 30px; font-weight: bold; padding: 10px; background-color: #2ecc71;")
        elif resultado == "NOK":
            self.label_resultado.setStyleSheet("font-size: 30px; font-weight: bold; padding: 10px; background-color: #e74c3c; color: white;")
        elif resultado in ("SIN_MASTER", "SIN_PIEZA"):
            self.label_resultado.setStyleSheet("font-size: 30px; font-weight: bold; padding: 10px; background-color: #f1c40f;")
        else:
            self.label_resultado.setStyleSheet("font-size: 30px; font-weight: bold; padding: 10px; background-color: #bdc3c7;")

        # El LED usa el veredicto ESTABLE para evitar parpadeos por ruido de
        # un frame individual. Una pieza NOK bloqueada dimensionalmente sigue
        # siendo NOK, por lo que el LED se mantiene rojo aunque el brazo no se mueva.
        if veredicto == "OK":
            estado_led = "OK"
        elif veredicto == "NOK":
            estado_led = "NOK"
        else:
            estado_led = "OFF"

        self.actualizar_led_fisico(estado_led)

    def actualizar_led_fisico(self, estado: str, forzar: bool = False) -> None:
        """Envía el estado del LED solo cuando cambia y el Arduino está conectado."""
        estado = estado.strip().upper()
        if estado not in {"OK", "NOK", "OFF"}:
            estado = "OFF"

        self.estado_led_deseado = estado

        if not self.brazo_conectado:
            return

        if not forzar and estado == self.ultimo_estado_led_solicitado:
            return

        self.ultimo_estado_led_solicitado = estado
        self.worker_brazo.solicitar_led(estado)


    def registrar_inspeccion(self):
        if not self.brazo_conectado:
            mensaje = "Conecta y activa el brazo antes de registrar una inspección."
            self.label_mensaje.setText(mensaje)
            QMessageBox.information(self, "Brazo desconectado", mensaje)
            return

        if self.brazo_ocupado:
            mensaje = "Espere: el brazo todavía está clasificando la pieza anterior."
            self.label_mensaje.setText(mensaje)
            return

        try:
            (
                ok,
                mensaje,
                veredicto,
                permitir_movimiento,
                motivo_bloqueo,
                id_inspeccion,
                inicio_ciclo,
                tiempo_inspeccion_s,
            ) = self.motor.registrar_inspeccion_actual()
            self.label_mensaje.setText(mensaje)

            if not ok:
                QMessageBox.information(self, "Registro", mensaje)
                return

            self.cargar_historial()

            # El registro exitoso debe devolver necesariamente un comando
            # de clasificación válido. Esta comprobación además permite a
            # Pylance/Pyright estrechar el tipo de veredicto de Optional[str]
            # a str antes de enviarlo al trabajador serial.
            if veredicto not in ("OK", "NOK"):
                mensaje_error = (
                    "La inspección se registró, pero no devolvió un "
                    "veredicto válido para el brazo."
                )
                self.label_mensaje.setText(mensaje_error)
                QMessageBox.warning(self, "Veredicto inválido", mensaje_error)
                return

            texto_inspeccion = (
                f"{tiempo_inspeccion_s:.3f}"
                if tiempo_inspeccion_s is not None
                else "--"
            )

            if not permitir_movimiento:
                self.label_tiempos.setText(
                    f"Tiempos | Inspección: {texto_inspeccion} s | "
                    "Brazo: N/A | Clasificación: N/A | Ciclo: N/A"
                )
                mensaje_final = (
                    f"{mensaje}. Movimiento del brazo bloqueado: "
                    f"{motivo_bloqueo}. Retira la pieza manualmente."
                )
                self.label_mensaje.setText(mensaje_final)
                QMessageBox.warning(
                    self,
                    "Movimiento bloqueado",
                    mensaje_final,
                )
                return

            self.id_inspeccion_pendiente = id_inspeccion
            self.inicio_ciclo_pendiente = inicio_ciclo
            self.tiempo_inspeccion_pendiente_s = tiempo_inspeccion_s

            self.label_tiempos.setText(
                f"Tiempos | Inspección: {texto_inspeccion} s | "
                "Brazo: midiendo... | Clasificación: midiendo... | Ciclo: midiendo..."
            )

            # Bloqueo inmediato para impedir un segundo registro antes de que
            # el hilo serial alcance a emitir el estado MOVIENDO.
            self.brazo_ocupado = True
            self.actualizar_disponibilidad_brazo()
            self.label_mensaje.setText(
                f"{mensaje}. Enviando clasificación {veredicto} al brazo..."
            )
            self.worker_brazo.solicitar_movimiento(veredicto)

        except Exception as e:
            QMessageBox.critical(self, "Error SQLite", str(e))

    def conectar_desconectar_brazo(self):
        if self.brazo_ocupado:
            QMessageBox.information(
                self,
                "Brazo ocupado",
                "No se puede desconectar mientras el brazo está en movimiento."
            )
            return

        if self.brazo_conectado:
            self.btn_conectar_brazo.setEnabled(False)
            self.label_mensaje.setText("Desconectando brazo...")
            self.worker_brazo.solicitar_desconexion()
            return

        respuesta = QMessageBox.question(
            self,
            "Activar brazo",
            "Verifica que el brazo esté físicamente en HOME y que el Monitor Serial "
            "de Arduino IDE esté cerrado.\n\n¿Conectar y enviar el comando A?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if respuesta != QMessageBox.StandardButton.Yes:
            return

        self.btn_conectar_brazo.setEnabled(False)
        self.label_mensaje.setText("Buscando Arduino y activando el brazo...")
        self.worker_brazo.solicitar_conexion()

    def enviar_home_brazo(self):
        if not self.brazo_conectado:
            QMessageBox.information(self, "Brazo", "El brazo no está conectado.")
            return
        if self.brazo_ocupado:
            QMessageBox.information(self, "Brazo", "El brazo ya está en movimiento.")
            return

        self.brazo_ocupado = True
        self.actualizar_disponibilidad_brazo()
        self.worker_brazo.solicitar_movimiento("H")

    def actualizar_estado_brazo(self, estado):
        self.estado_brazo = estado
        self.brazo_conectado = estado in ("LISTO", "MOVIENDO")
        self.brazo_ocupado = estado == "MOVIENDO"

        estilos = {
            "DESCONECTADO": "background-color: #bdc3c7; color: #222;",
            "CONECTANDO": "background-color: #f1c40f; color: #222;",
            "LISTO": "background-color: #2ecc71; color: #111;",
            "MOVIENDO": "background-color: #3498db; color: white;",
            "ERROR": "background-color: #e74c3c; color: white;",
        }

        self.label_estado_brazo.setText(f"Brazo: {estado}")
        self.label_estado_brazo.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 6px; "
            + estilos.get(estado, "background-color: #bdc3c7;")
        )
        self.actualizar_disponibilidad_brazo()

    def actualizar_disponibilidad_brazo(self):
        conectado_y_libre = self.brazo_conectado and not self.brazo_ocupado

        self.btn_registrar.setEnabled(conectado_y_libre)
        self.btn_home_brazo.setEnabled(conectado_y_libre)

        if hasattr(self, "shortcut_registrar"):
            self.shortcut_registrar.setEnabled(conectado_y_libre)

        if self.estado_brazo == "CONECTANDO":
            self.btn_conectar_brazo.setEnabled(False)
        elif self.brazo_ocupado:
            self.btn_conectar_brazo.setEnabled(False)
        else:
            self.btn_conectar_brazo.setEnabled(True)

        self.btn_conectar_brazo.setText(
            "Desconectar brazo" if self.brazo_conectado else "Conectar y activar brazo"
        )

    def mostrar_mensaje_brazo(self, mensaje):
        # También se imprime para conservar el registro completo del protocolo.
        print(mensaje)

    def brazo_conectado_correctamente(self, puerto):
        self.label_puerto_brazo.setText(f"Puerto del brazo: {puerto}")
        self.label_mensaje.setText("Brazo conectado, activado y listo.")
        self.ultimo_estado_led_solicitado = None
        self.actualizar_led_fisico(self.estado_led_deseado, forzar=True)

    def brazo_desconectado_correctamente(self):
        self.label_puerto_brazo.setText("Puerto del brazo: detección automática")
        self.label_mensaje.setText("Brazo desconectado.")
        self.ultimo_estado_led_solicitado = None

    def movimiento_brazo_iniciado(self, comando):
        self.label_mensaje.setText(f"Brazo ejecutando comando {comando}...")
        if comando in ("OK", "NOK") and self.id_inspeccion_pendiente is not None:
            try:
                self.motor.marcar_movimiento_iniciado(
                    self.id_inspeccion_pendiente,
                    comando,
                )
            except Exception as exc:
                print(f"No se pudo marcar movimiento iniciado: {exc}")

    def movimiento_brazo_terminado(self, comando, tiempo_brazo_s):
        if comando in ("OK", "NOK"):
            if self.inicio_ciclo_pendiente is not None:
                tiempo_total_ciclo_s = max(
                    0.0,
                    time.monotonic() - self.inicio_ciclo_pendiente,
                )
            else:
                tiempo_total_ciclo_s = None

            if (
                self.tiempo_inspeccion_pendiente_s is not None
                and tiempo_brazo_s is not None
            ):
                tiempo_clasificacion_s = (
                    self.tiempo_inspeccion_pendiente_s + tiempo_brazo_s
                )
            else:
                tiempo_clasificacion_s = None

            try:
                self.motor.finalizar_movimiento_inspeccion(
                    self.id_inspeccion_pendiente,
                    comando,
                    tiempo_brazo_s,
                    tiempo_total_ciclo_s,
                    self.tiempo_inspeccion_pendiente_s,
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Tiempos",
                    f"El brazo terminó, pero no se pudieron guardar los tiempos: {exc}",
                )

            t_ins = (
                f"{self.tiempo_inspeccion_pendiente_s:.3f}"
                if self.tiempo_inspeccion_pendiente_s is not None
                else "--"
            )
            t_brazo = f"{tiempo_brazo_s:.3f}" if tiempo_brazo_s is not None else "--"
            t_clas = (
                f"{tiempo_clasificacion_s:.3f}"
                if tiempo_clasificacion_s is not None
                else "--"
            )
            t_total = (
                f"{tiempo_total_ciclo_s:.3f}"
                if tiempo_total_ciclo_s is not None
                else "--"
            )

            self.label_tiempos.setText(
                f"Tiempos | Inspección: {t_ins} s | Brazo: {t_brazo} s | "
                f"Clasificación: {t_clas} s | Ciclo: {t_total} s"
            )
            self.label_mensaje.setText(
                f"Clasificación {comando} terminada. Coloque la siguiente pieza."
            )
            self.cargar_historial()

            self.id_inspeccion_pendiente = None
            self.inicio_ciclo_pendiente = None
            self.tiempo_inspeccion_pendiente_s = None
        else:
            self.label_mensaje.setText(
                f"El brazo llegó a HOME. Tiempo de movimiento: {tiempo_brazo_s:.3f} s."
            )

    def mostrar_error_brazo(self, mensaje):
        if self.id_inspeccion_pendiente is not None:
            try:
                self.motor.registrar_error_movimiento(
                    self.id_inspeccion_pendiente,
                    mensaje,
                )
            except Exception as exc:
                print(f"No se pudo guardar el error del movimiento: {exc}")

        self.id_inspeccion_pendiente = None
        self.inicio_ciclo_pendiente = None
        self.tiempo_inspeccion_pendiente_s = None

        self.brazo_conectado = False
        self.brazo_ocupado = False
        self.ultimo_estado_led_solicitado = None
        self.actualizar_disponibilidad_brazo()
        self.label_mensaje.setText(f"Error del brazo: {mensaje}")
        self.cargar_historial()
        QMessageBox.critical(self, "Error del brazo", mensaje)

    def toggle_registro(self):
        habilitado = self.motor.toggle_registro()
        if habilitado:
            self.btn_pausar.setText("Pausar registro")
            self.label_mensaje.setText("Registro habilitado.")
        else:
            self.btn_pausar.setText("Reanudar registro")
            self.label_mensaje.setText("Registro pausado.")

    def capturar_master(self):
        ok, mensaje = self.motor.capturar_master_actual()
        self.label_mensaje.setText(mensaje)
        if not ok:
            QMessageBox.information(self, "Master", mensaje)

    def reiniciar_master(self):
        mensaje = self.motor.reiniciar_master_pendiente()
        self.label_mensaje.setText(mensaje)

    def cargar_historial(self):
        try:
            filas = self.motor.cargar_historial(limite=50)
            self.tabla.setRowCount(len(filas))

            for fila_idx, fila in enumerate(filas):
                for col_idx, valor in enumerate(fila):
                    item = QTableWidgetItem("" if valor is None else str(valor))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                    if col_idx == 2:
                        if valor == "OK":
                            item.setBackground(Qt.GlobalColor.green)
                        elif valor == "NOK":
                            item.setBackground(Qt.GlobalColor.red)
                        else:
                            item.setBackground(Qt.GlobalColor.yellow)

                    self.tabla.setItem(fila_idx, col_idx, item)

            self.actualizar_indicadores_operacion()
        except Exception as e:
            self.label_mensaje.setText(f"No se pudo cargar historial: {e}")

    def actualizar_indicadores_operacion(self):
        """Actualiza las tarjetas de operación con datos agregados de SQLite."""
        try:
            datos = self.motor.cargar_indicadores_operacion()

            self.label_ind_total.setText(
                f"Inspecciones totales\n{datos['inspecciones_totales']}"
            )
            self.label_ind_ok.setText(f"OK\n{datos['ok']}")
            self.label_ind_nok.setText(f"NOK\n{datos['nok']}")

            promedio = datos.get("tiempo_promedio_ciclo_s")
            if promedio is None:
                texto_promedio = "-- s"
            else:
                texto_promedio = f"{promedio:.2f} s"

            self.label_ind_tiempo.setText(
                f"Tiempo promedio del ciclo\n{texto_promedio}"
            )
        except Exception as exc:
            print(f"No se pudieron actualizar los indicadores: {exc}")

    def mostrar_error(self, mensaje):
        self.label_mensaje.setText(f"Error cámara/procesamiento: {mensaje}")

    def closeEvent(self, event):
        if self.brazo_ocupado:
            respuesta = QMessageBox.question(
                self,
                "Brazo en movimiento",
                "El brazo todavía está ejecutando una trayectoria. Cerrar la GUI no "
                "detendrá el movimiento autónomo del Arduino.\n\n¿Cerrar de todos modos?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if respuesta != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        try:
            self.worker.detener()
            self.motor.liberar()
        except Exception:
            pass

        try:
            self.worker_brazo.detener()
        except Exception:
            pass

        event.accept()


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaInspeccion()
    ventana.show()
    sys.exit(app.exec())