import cv2
import numpy as np
import os
import sqlite3
import time
from datetime import datetime
from collections import deque

# ==============================
# CONFIGURACIÓN
# ==============================

CAMERA_INDEX = 2

ARCHIVO_HOMOGRAFIA = "calibracion/resultados/homografia_jig_corregida.npz"
ARCHIVO_SHAPE_MASTER = "calibracion/resultados/shape_master.npz"
ARCHIVO_LOG_CALIBRACION = "calibracion/resultados/log_scores_shape.csv"
# Ruta base del proyecto tomando como referencia este archivo .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCHIVO_BD_INSPECCIONES = os.path.join(
    BASE_DIR,
    "base_datos",
    "inspecciones.db"
)

# Escala de la imagen rectificada
# 10 px/mm significa que 1 mm en el plano real se representa con 10 píxeles
PX_PER_MM = 10

# Agujeros esperados
AGUJEROS_ESPERADOS = [
    {"id": "C1", "nominal": 20.0, "tol": 1.0},
    {"id": "C2", "nominal": 7.0,  "tol": 1.0},
    {"id": "C3", "nominal": 7.0,  "tol": 1.0},
]

# Ranuras esperadas
RANURAS_ESPERADAS = [
    {
        "id": "R1",
        "tipo": "recta",
        "cantidad_esperada": 1,
        "largo_nominal": 21.0,
        "ancho_nominal": 6.0,
        "tol": 1.0
    },
    {
        "id": "R3",
        "tipo": "cuadrada",
        "cantidad_esperada": 1,
        "largo_nominal": 7.0,
        "ancho_nominal": 7.0,
        "tol": 1.0
    }
]

# Contorno exterior esperado según se observa en la cámara
CONTORNO_ESPERADO = {
    "id": "P1",
    "lado_mayor_nominal": 70.0,
    "lado_menor_nominal": 40.0,
    "tol": 1.0
}

# Límite independiente de seguridad para la trayectoria fija del brazo.
# Si el error absoluto de CUALQUIERA de los lados es de 6 mm o más, la
# inspección se conserva como NOK y se registra, pero no se envía movimiento.
MARGEN_MAXIMO_MOVIMIENTO_BRAZO_MM = 5.0

# ==============================
# FILTRO DE PLAUSIBILIDAD (descarta manos / objetos ajenos)
# ==============================

# El filtro de área mínima (area_pieza_min) solo pone un piso: cualquier
# objeto grande en la escena (una mano metida para retirar/colocar la
# pieza) también lo pasa, se evalúa como si fuera la pieza, sale NOK
# por no tener la forma/dimensiones correctas, y se guarda en SQLite
# como si fuera una pieza real inspeccionada.
#
# Este filtro de IDENTIDAD es deliberadamente más tolerante que la
# inspección dimensional, pero mucho más estricto que el filtro anterior.
# Su objetivo es separar dos decisiones distintas:
#   1) ¿El objeto pertenece a la familia de la placa?  -> pieza detectada
#   2) ¿La placa cumple las tolerancias?               -> OK / NOK
#
# Una placa reconocible puede seguir siendo NOK por dimensiones, agujeros,
# ranuras, posición o forma. En cambio, una mano, herramienta u objeto cuya
# geometría exterior no se parece a la placa se reporta como SIN_PIEZA y
# nunca se registra ni activa el brazo.
FILTRO_PIEZA_MARGEN_LADO_MAYOR_MM = 10.0
FILTRO_PIEZA_MARGEN_LADO_MENOR_MM = 8.0
FILTRO_PIEZA_RELLENO_MIN = 0.70
FILTRO_PIEZA_AREA_REL_MIN = 0.55
FILTRO_PIEZA_AREA_REL_MAX = 1.30
FILTRO_PIEZA_CENTROIDE_TOL_MM = 8.0

# Área exterior ideal
AREA_EXTERIOR_ESPERADA_MM2 = 70.0 * 40.0

# Este valor queda solo como diagnóstico visual, no como rechazo principal
TOL_AREA_EXTERIOR_PORC = 1.0

# Relación mínima entre área real y área del rectángulo envolvente.
# Si la pieza buena sale NOK, baja a 0.980.
# Si la esquina cortada sale OK, sube a 0.990.
FACTOR_RELLENO_RECT_MIN = 0.95

# Evaluación de esquinas
# Se revisa si las 4 esquinas del rectángulo están ocupadas por pieza.
ESQUINA_PATCH_MM = 4.0
ESQUINA_RELLENO_MIN = 0.65

# Vértices exteriores: por ahora solo diagnóstico, no rechazo obligatorio
VERTICES_EXTERIOR_ESPERADOS = 4
USAR_VERTICES_EN_CONTORNO = True

# Orientación esperada de la pieza
ORIENTACION_ESPERADA = {
    "id": "O1",
    "angulo_nominal": 0.0,
    "tol": 5.0
}

# Centroide esperado
CENTROIDE_TOL_MM = 1.0

# ==============================
# SHAPE MATCHING
# ==============================

USAR_SHAPE_MATCHING = False

# Al inicio déjalo en False para no volver inestable la detección.
# Después de probar, puedes cambiarlo a True para validar también ranuras por forma.
USAR_SHAPE_RANURAS = False

# Umbrales iniciales.
# Mientras menor sea el score, más parecida es la pieza a la master.
UMBRAL_SHAPE_MASCARA = 0.0050
UMBRAL_SHAPE_EXTERIOR = 0.0080
UMBRAL_SHAPE_RANURA_RECTA = 0.0080
UMBRAL_SHAPE_RANURA_CUADRADA = 0.0080

# Número de piezas OK que se capturan para construir la master.
# Se promedian entre sí (máscara) o se comparan todas contra la pieza
# actual tomando el mejor match (contornos), para que la referencia
# tolere la variación normal entre piezas buenas.
NUM_MUESTRAS_MASTER = 5

# ==============================
# VALIDACIÓN DE POSICIÓN DE FEATURES
# ==============================

# El Shape Matching por momentos de Hu es un descriptor GLOBAL: es
# insensible a que un agujero/ranura pequeño se haya movido de sitio
# dentro de la pieza (el área y perímetro totales casi no cambian).
# Por eso, además de validar diámetro/dimensiones nominales, se valida
# también la POSICIÓN (X,Y en mm) de cada agujero y ranura contra la
# posición nominal, calculada automáticamente como el promedio de las
# posiciones observadas en las muestras de la shape master.
USAR_VALIDACION_POSICION = True

POSICION_AGUJERO_TOL_MM = 1.0
POSICION_RANURA_TOL_MM = 1.0

# Distancia máxima para relacionar un círculo detectado con la posición
# nominal de C1, C2 o C3. Debe ser mayor que la tolerancia real de posición
# para permitir que un agujero desplazado siga identificado con su ID y sea
# reportado como "fuera de posición", pero no tan grande como para asociarlo
# con la ubicación de otro agujero.
DISTANCIA_MAX_ASIGNACION_AGUJERO_MM = 12.0

# Convención usada ÚNICAMENTE al construir una nueva Shape Master, cuando
# todavía no existen posiciones nominales. C1 se reconoce por ser el círculo
# cuyo diámetro está más cerca de 20 mm. Para C2 y C3 (ambos de 7 mm), se usa
# su posición: "auto" elige el eje donde están más separados; C2 queda como
# el de coordenada menor (izquierda o arriba) y C3 como el de coordenada mayor
# (derecha o abajo). Si en tu plano físico están nombrados al revés, cambia
# C2_ORDEN_ASCENDENTE a False y vuelve a capturar la Shape Master.
EJE_ORDEN_C2_C3 = "auto"       # "auto", "x" o "y"
C2_ORDEN_ASCENDENTE = True

# ==============================
# ESTABILIZACIÓN TEMPORAL
# ==============================

# El veredicto de cada frame individual puede parpadear por ruido de
# cámara/umbralización (un pixel de más o de menos en el borde mueve
# el score de Shape Matching lo suficiente para cruzar el umbral).
# En vez de confiar en un solo frame, se exige que los últimos N
# resultados coincidan antes de dar un veredicto ESTABLE.
FRAMES_CONSECUTIVOS_REQUERIDOS = 8

# ==============================
# CONTROL DE REGISTRO DE PIEZAS (MANUAL)
# ==============================

# El registro ya NO es automático por estabilidad física: el operador
# decide cuándo inspeccionar presionando una tecla. Esto da control
# total sobre el momento del registro y sirve como disparador limpio
# para mover un brazo robótico únicamente cuando se realiza una
# inspección real (ver la función disparar_accion_externa más abajo).
TECLA_REGISTRAR = ord("i")

# Pequeño cooldown SOLO como protección contra rebote de tecla
# (key-repeat del sistema operativo si se mantiene presionada), no
# como mecanismo de estabilidad — esa decisión ahora es tuya.
COOLDOWN_BOTON_SEG = 1.0

# Pausa manual del registro. Útil mientras ajustas trackbars o mueves
# el brazo dentro del área de inspección: con esto en False, la tecla
# de registro no hace nada aunque la presiones.
REGISTRO_HABILITADO_MANUAL_DEFAULT = True

# ==============================
# EVIDENCIA FOTOGRÁFICA (SOLO PIEZAS NOK)
# ==============================

# Guardar una imagen por cada pieza (buena o mala) llena el disco
# rápido en producción continua. Como las piezas OK no necesitan
# revisión visual, solo se guarda evidencia de las que salen NOK —
# eso es justo lo que necesitas para diagnosticar fallas después.
GUARDAR_IMAGENES_SOLO_NOK = True

CARPETA_EVIDENCIAS = os.path.join(BASE_DIR, "evidencias")

# ==============================
# PANEL DE RESULTADOS

# ==============================

PANEL_ANCHO = 950
PANEL_ALTO = 1100
PANEL_LINEA = 30
PANEL_ESCALA = 0.50
PANEL_MARGEN_SUP = 45
PANEL_MARGEN_INF = 35
PANEL_MARGEN_IZQ = 20


def cargar_float(datos, clave, valor_default):
    if clave in datos.files:
        return float(np.array(datos[clave]).reshape(-1)[0])
    return valor_default


def evaluar_medida(valor, nominal, tol):
    lim_inf = nominal - tol
    lim_sup = nominal + tol

    if lim_inf <= valor <= lim_sup:
        return "OK"
    else:
        return "NOK"


def evaluar_diametro(diametro_mm, nominal, tol):
    return evaluar_medida(diametro_mm, nominal, tol)


def crear_matriz_rectificacion(H_img_to_mm, px_per_mm):
    """
    Convierte la homografía imagen->mm a imagen->imagen_rectificada_px.
    """
    S = np.array([
        [px_per_mm, 0, 0],
        [0, px_per_mm, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    H_img_to_rect = S @ H_img_to_mm
    return H_img_to_rect


def calcular_circularidad(contorno):
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)

    if perimetro == 0:
        return 0

    circularidad = (4 * np.pi * area) / (perimetro ** 2)
    return circularidad


def obtener_diametro_equivalente_mm(area_px, px_per_mm):
    """
    Calcula el diámetro equivalente a partir del área:
    A = pi * r^2
    D = 2 * sqrt(A/pi)
    """
    area_mm2 = area_px / (px_per_mm ** 2)
    diametro_mm = 2 * np.sqrt(area_mm2 / np.pi)
    return diametro_mm


def normalizar_angulo_grados(angulo):
    """
    Normaliza un ángulo para dejarlo entre -45 y 45 grados.
    """
    while angulo > 45:
        angulo -= 90

    while angulo < -45:
        angulo += 90

    return angulo


def obtener_angulo_lado_mayor(box):
    """
    Calcula el ángulo del lado mayor del rectángulo respecto al eje X.
    Devuelve un ángulo normalizado entre -45 y 45 grados.
    """
    puntos = box.astype(np.float32)
    lados = []

    for i in range(4):
        p1 = puntos[i]
        p2 = puntos[(i + 1) % 4]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        longitud = np.sqrt(dx**2 + dy**2)
        angulo = np.degrees(np.arctan2(dy, dx))

        lados.append((longitud, angulo))

    lado_mayor = max(lados, key=lambda x: x[0])
    angulo_lado_mayor = lado_mayor[1]

    return normalizar_angulo_grados(angulo_lado_mayor)


def evaluar_relleno_esquinas(contorno_pieza, box, image_shape, px_per_mm):
    """
    Evalúa si las 4 esquinas del rectángulo exterior están rellenas por la pieza.
    Esto ayuda a detectar una esquina cortada, aunque el minAreaRect siga midiendo bien.

    Se crea un pequeño parche cuadrado hacia adentro en cada esquina del rectángulo.
    Si el parche no está suficientemente ocupado por la máscara de la pieza, se marca NOK.
    """
    mask_contorno = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask_contorno, [contorno_pieza], -1, 255, thickness=-1)

    box_float = box.astype(np.float32)

    patch_px = int(round(ESQUINA_PATCH_MM * px_per_mm))
    patch_px = max(patch_px, 5)

    rellenos = []

    for i in range(4):
        p = box_float[i]
        p_prev = box_float[(i - 1) % 4]
        p_next = box_float[(i + 1) % 4]

        v1 = p_prev - p
        v2 = p_next - p

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            rellenos.append(0.0)
            continue

        u1 = v1 / norm1
        u2 = v2 / norm2

        # Parche hacia adentro desde la esquina
        q0 = p
        q1 = p + u1 * patch_px
        q2 = p + u1 * patch_px + u2 * patch_px
        q3 = p + u2 * patch_px

        patch_poly = np.array([q0, q1, q2, q3], dtype=np.int32)

        mask_patch = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask_patch, patch_poly, 255)

        area_patch = cv2.countNonZero(mask_patch)

        if area_patch == 0:
            relleno = 0.0
        else:
            area_ocupada = cv2.countNonZero(cv2.bitwise_and(mask_contorno, mask_patch))
            relleno = area_ocupada / area_patch

        rellenos.append(relleno)

    relleno_min = min(rellenos) if rellenos else 0.0

    if relleno_min >= ESQUINA_RELLENO_MIN:
        estado_esquinas = "OK"
    else:
        estado_esquinas = "NOK"

    return relleno_min, rellenos, estado_esquinas


def evaluar_contorno_exterior(contorno_pieza, px_per_mm, image_shape):
    """
    Evalúa el contorno exterior de la pieza:
    - lado mayor
    - lado menor
    - forma exterior
    - esquinas
    - orientación
    - centroide
    """

    rect = cv2.minAreaRect(contorno_pieza)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    (cx_px, cy_px), (w_px, h_px), angle_raw = rect

    lado_a_mm = w_px / px_per_mm
    lado_b_mm = h_px / px_per_mm

    lado_mayor_mm = max(lado_a_mm, lado_b_mm)
    lado_menor_mm = min(lado_a_mm, lado_b_mm)

    # Área real del contorno exterior
    area_contorno_px = cv2.contourArea(contorno_pieza)
    area_contorno_mm2 = area_contorno_px / (px_per_mm ** 2)

    # Área del rectángulo envolvente medido
    area_rectangulo_mm2 = lado_mayor_mm * lado_menor_mm

    # Error porcentual respecto al área ideal.
    # Se deja como dato informativo.
    error_area_porc = (
        abs(area_contorno_mm2 - AREA_EXTERIOR_ESPERADA_MM2)
        / AREA_EXTERIOR_ESPERADA_MM2
        * 100.0
    )

    # Relación de llenado respecto al rectángulo envolvente
    if area_rectangulo_mm2 > 0:
        factor_relleno_rect = area_contorno_mm2 / area_rectangulo_mm2
    else:
        factor_relleno_rect = 0.0

    if factor_relleno_rect >= FACTOR_RELLENO_RECT_MIN:
        estado_forma_exterior = "OK"
    else:
        estado_forma_exterior = "NOK"

    # Vértices exteriores
    perimetro_ext = cv2.arcLength(contorno_pieza, True)
    epsilon_ext = 0.01 * perimetro_ext
    approx_ext = cv2.approxPolyDP(contorno_pieza, epsilon_ext, True)
    vertices_exterior = len(approx_ext)

    if vertices_exterior == VERTICES_EXTERIOR_ESPERADOS:
        estado_vertices_exterior = "OK"
    else:
        estado_vertices_exterior = "NOK"

    # Evaluación de esquinas
    relleno_min_esquina, rellenos_esquinas, estado_esquinas = evaluar_relleno_esquinas(
        contorno_pieza,
        box,
        image_shape,
        px_per_mm
    )

    # Evaluar dimensiones
    estado_lado_mayor = evaluar_medida(
        lado_mayor_mm,
        CONTORNO_ESPERADO["lado_mayor_nominal"],
        CONTORNO_ESPERADO["tol"]
    )

    estado_lado_menor = evaluar_medida(
        lado_menor_mm,
        CONTORNO_ESPERADO["lado_menor_nominal"],
        CONTORNO_ESPERADO["tol"]
    )

    # Orientación
    angulo = obtener_angulo_lado_mayor(box)

    estado_orientacion = evaluar_medida(
        angulo,
        ORIENTACION_ESPERADA["angulo_nominal"],
        ORIENTACION_ESPERADA["tol"]
    )

    # Centroide por momentos
    M = cv2.moments(contorno_pieza)

    if M["m00"] != 0:
        cx_mom_px = int(M["m10"] / M["m00"])
        cy_mom_px = int(M["m01"] / M["m00"])
    else:
        cx_mom_px = int(cx_px)
        cy_mom_px = int(cy_px)

    cx_mm = cx_mom_px / px_per_mm
    cy_mm = cy_mom_px / px_per_mm

    centroide_esperado_x = W_JIG_MM / 2.0
    centroide_esperado_y = H_JIG_MM / 2.0

    error_centroide_x = abs(cx_mm - centroide_esperado_x)
    error_centroide_y = abs(cy_mm - centroide_esperado_y)

    if error_centroide_x <= CENTROIDE_TOL_MM and error_centroide_y <= CENTROIDE_TOL_MM:
        estado_centroide = "OK"
    else:
        estado_centroide = "NOK"

    # Estado del contorno exterior
    condiciones_contorno = [
        estado_lado_mayor == "OK",
        estado_lado_menor == "OK",
        estado_forma_exterior == "OK",
        estado_esquinas == "OK"
    ]

    if USAR_VERTICES_EN_CONTORNO:
        condiciones_contorno.append(estado_vertices_exterior == "OK")

    if all(condiciones_contorno):
        estado_contorno = "OK"
    else:
        estado_contorno = "NOK"

    return {
        "box": box,
        "lado_mayor_mm": lado_mayor_mm,
        "lado_menor_mm": lado_menor_mm,
        "estado_lado_mayor": estado_lado_mayor,
        "estado_lado_menor": estado_lado_menor,
        "estado_contorno": estado_contorno,
        "angulo": angulo,
        "estado_orientacion": estado_orientacion,
        "cx_px": cx_mom_px,
        "cy_px": cy_mom_px,
        "cx_mm": cx_mm,
        "cy_mm": cy_mm,
        "centroide_esperado_x": centroide_esperado_x,
        "centroide_esperado_y": centroide_esperado_y,
        "error_centroide_x": error_centroide_x,
        "error_centroide_y": error_centroide_y,
        "estado_centroide": estado_centroide,
        "area_contorno_mm2": area_contorno_mm2,
        "area_esperada_mm2": AREA_EXTERIOR_ESPERADA_MM2,
        "error_area_porc": error_area_porc,
        "factor_relleno_rect": factor_relleno_rect,
        "estado_forma_exterior": estado_forma_exterior,
        "vertices_exterior": vertices_exterior,
        "estado_vertices_exterior": estado_vertices_exterior,
        "relleno_min_esquina": relleno_min_esquina,
        "rellenos_esquinas": rellenos_esquinas,
        "estado_esquinas": estado_esquinas
    }


def es_objeto_plausible_como_pieza(resultado_exterior):
    """Determina si el objeto pertenece a la familia geométrica de la placa.

    Este filtro NO decide si la pieza es buena o mala. Solo impide que un
    objeto ajeno llegue a clasificarse como NOK y active el brazo.

    Se exige simultáneamente:
    - largo y ancho dentro de un margen amplio respecto a 70 x 40 mm;
    - contorno exterior suficientemente rectangular/relleno;
    - área exterior razonable respecto al área nominal;
    - centroide dentro de una zona amplia alrededor del jig.

    Una placa con agujeros o ranuras defectuosos seguirá pasando este filtro
    y después será evaluada normalmente como NOK.
    """
    if resultado_exterior is None:
        return False

    error_lado_mayor = abs(
        resultado_exterior["lado_mayor_mm"]
        - CONTORNO_ESPERADO["lado_mayor_nominal"]
    )
    error_lado_menor = abs(
        resultado_exterior["lado_menor_mm"]
        - CONTORNO_ESPERADO["lado_menor_nominal"]
    )

    dimensiones_plausibles = (
        error_lado_mayor <= FILTRO_PIEZA_MARGEN_LADO_MAYOR_MM
        and error_lado_menor <= FILTRO_PIEZA_MARGEN_LADO_MENOR_MM
    )

    relleno_plausible = (
        resultado_exterior["factor_relleno_rect"]
        >= FILTRO_PIEZA_RELLENO_MIN
    )

    area_relativa = (
        resultado_exterior["area_contorno_mm2"]
        / AREA_EXTERIOR_ESPERADA_MM2
    )
    area_plausible = (
        FILTRO_PIEZA_AREA_REL_MIN
        <= area_relativa
        <= FILTRO_PIEZA_AREA_REL_MAX
    )

    centroide_plausible = (
        resultado_exterior["error_centroide_x"]
        <= FILTRO_PIEZA_CENTROIDE_TOL_MM
        and resultado_exterior["error_centroide_y"]
        <= FILTRO_PIEZA_CENTROIDE_TOL_MM
    )

    return (
        dimensiones_plausibles
        and relleno_plausible
        and area_plausible
        and centroide_plausible
    )


def evaluar_seguridad_movimiento_brazo(resultado_exterior):
    """Evalúa si la trayectoria fija puede ejecutarse con seguridad.

    Esta función NO cambia el veredicto de inspección. Una placa fuera de este
    rango sigue siendo NOK y puede guardarse en SQLite, pero el brazo queda
    bloqueado para que el operador la retire manualmente.

    El límite es inclusivo: una desviación de exactamente 6.0 mm ya bloquea.
    """
    if resultado_exterior is None:
        return {
            "permitir_movimiento": False,
            "motivo": "No se detectó el contorno exterior.",
            "error_lado_mayor_mm": None,
            "error_lado_menor_mm": None,
        }

    lado_mayor = float(resultado_exterior["lado_mayor_mm"])
    lado_menor = float(resultado_exterior["lado_menor_mm"])

    error_lado_mayor = abs(
        lado_mayor - CONTORNO_ESPERADO["lado_mayor_nominal"]
    )
    error_lado_menor = abs(
        lado_menor - CONTORNO_ESPERADO["lado_menor_nominal"]
    )

    # Se permite únicamente cuando ambos errores son menores a 6 mm.
    permitir_movimiento = (
        error_lado_mayor < MARGEN_MAXIMO_MOVIMIENTO_BRAZO_MM
        and error_lado_menor < MARGEN_MAXIMO_MOVIMIENTO_BRAZO_MM
    )

    if permitir_movimiento:
        motivo = "Dimensiones dentro del rango seguro de agarre."
    else:
        causas = []

        if error_lado_mayor >= MARGEN_MAXIMO_MOVIMIENTO_BRAZO_MM:
            causas.append(
                "largo fuera del rango seguro "
                f"({lado_mayor:.2f} mm; nominal "
                f"{CONTORNO_ESPERADO['lado_mayor_nominal']:.2f} mm)"
            )

        if error_lado_menor >= MARGEN_MAXIMO_MOVIMIENTO_BRAZO_MM:
            causas.append(
                "ancho fuera del rango seguro "
                f"({lado_menor:.2f} mm; nominal "
                f"{CONTORNO_ESPERADO['lado_menor_nominal']:.2f} mm)"
            )

        motivo = "; ".join(causas) or "Dimensiones fuera del rango seguro."

    return {
        "permitir_movimiento": permitir_movimiento,
        "motivo": motivo,
        "error_lado_mayor_mm": error_lado_mayor,
        "error_lado_menor_mm": error_lado_menor,
    }


def obtener_primera_ranura_por_tipo(ranuras_detectadas, tipo):
    """
    Devuelve la primera ranura detectada de cierto tipo.
    tipo puede ser: 'recta' o 'cuadrada'.
    """
    for r in ranuras_detectadas:
        if r["tipo"] == tipo:
            return r

    return None


def promediar_mascaras(lista_mascaras):
    """
    Promedia varias máscaras binarias pixel a pixel y vuelve a binarizar
    el resultado. Esto suaviza el ruido de borde de cada captura
    individual (umbralización, iluminación) y produce una máscara de
    referencia más estable que cualquier muestra suelta.

    Un pixel queda en la máscara final si aparece "encendido" en al
    menos la mitad de las muestras.
    """
    pila = np.stack(lista_mascaras, axis=0).astype(np.float32)
    promedio = pila.mean(axis=0)
    mascara_final = (promedio >= 127.5).astype(np.uint8) * 255
    return mascara_final


def promediar_posiciones(lista_posiciones):
    """
    Recibe una lista (una entrada por muestra) donde cada entrada es
    (x_mm, y_mm) o None si esa muestra no tenía el feature detectado.
    Devuelve (posicion_nominal, dispersion_mm) usando solo las muestras
    válidas, o (None, None) si ninguna muestra lo detectó.

    dispersion_mm es la distancia máxima de cualquier muestra respecto
    al promedio: sirve como diagnóstico de qué tan consistente fue la
    posición del feature entre las piezas usadas para la master (si
    sale muy alta, probablemente una de las capturas estaba mal o el
    fixture no sostiene la pieza de forma repetible).
    """
    validas = [p for p in lista_posiciones if p is not None]

    if not validas:
        return None, None

    x_prom = sum(p[0] for p in validas) / len(validas)
    y_prom = sum(p[1] for p in validas) / len(validas)

    dispersion = max(
        ((p[0] - x_prom) ** 2 + (p[1] - y_prom) ** 2) ** 0.5
        for p in validas
    )

    return (x_prom, y_prom), dispersion


def guardar_shape_master_multiple(archivo, muestras):
    """
    Guarda la referencia master para Shape Matching a partir de VARIAS
    muestras de piezas buenas:
    - máscara consenso (promedio binarizado de todas las máscaras)
    - lista de contornos exteriores, una por muestra
    - lista de contornos de ranura recta / cuadrada, una por muestra
      (puede haber None si esa muestra no tenía la ranura detectada)
    - posición nominal (X,Y en mm) de cada agujero esperado y de cada
      ranura, promediada entre las muestras que sí las detectaron

    Las listas de contornos se guardan con dtype=object porque cada
    contorno puede tener un número distinto de puntos.
    """

    os.makedirs(os.path.dirname(archivo), exist_ok=True)

    mask_master_promedio = promediar_mascaras(muestras["masks"])

    contornos_exterior = np.array(muestras["contornos_exterior"], dtype=object)
    contornos_r1 = np.array(muestras["contornos_r1"], dtype=object)
    contornos_r3 = np.array(muestras["contornos_r3"], dtype=object)

    # Posición nominal por cada agujero esperado (C1, C2, C3 en el
    # mismo orden que AGUJEROS_ESPERADOS: por diámetro descendente).
    posiciones_agujeros_nominal = []

    for idx in range(len(AGUJEROS_ESPERADOS)):
        posiciones_muestra = [
            muestra[idx] if idx < len(muestra) else None
            for muestra in muestras["posiciones_agujeros"]
        ]
        nominal, dispersion = promediar_posiciones(posiciones_muestra)
        posiciones_agujeros_nominal.append(nominal)

        id_ag = AGUJEROS_ESPERADOS[idx]["id"]
        if nominal is None:
            print(f"AVISO: ningún muestreo detectó el agujero {id_ag}; sin posición nominal para él.")
        else:
            texto_dispersion = (
                f"{dispersion:.2f}" if dispersion is not None else "--"
            )
            print(
                f"Posición nominal {id_ag}: X={nominal[0]:.2f} Y={nominal[1]:.2f} mm "
                f"(dispersión entre muestras: {texto_dispersion} mm)"
            )
            if dispersion is not None and dispersion > POSICION_AGUJERO_TOL_MM:
                print(
                    f"  AVISO: la dispersión ({dispersion:.2f} mm) ya supera la tolerancia "
                    f"configurada ({POSICION_AGUJERO_TOL_MM} mm). Revisa que el fixture "
                    "sostenga la pieza de forma repetible o sube POSICION_AGUJERO_TOL_MM."
                )

    posicion_r1_nominal, dispersion_r1 = promediar_posiciones(muestras["posiciones_r1"])
    posicion_r3_nominal, dispersion_r3 = promediar_posiciones(muestras["posiciones_r3"])

    for nombre, nominal, dispersion in [
        ("R1", posicion_r1_nominal, dispersion_r1),
        ("R3", posicion_r3_nominal, dispersion_r3),
    ]:
        if nominal is None:
            print(f"AVISO: ningún muestreo detectó la ranura {nombre}; sin posición nominal para ella.")
        else:
            texto_dispersion = (
                f"{dispersion:.2f}" if dispersion is not None else "--"
            )
            print(
                f"Posición nominal {nombre}: X={nominal[0]:.2f} Y={nominal[1]:.2f} mm "
                f"(dispersión entre muestras: {texto_dispersion} mm)"
            )

    # np.savez no acepta None directamente dentro de un array numérico;
    # se guardan como object arrays para poder representar "sin dato".
    np.savez_compressed(
        archivo,
        mask_master_promedio=mask_master_promedio,
        num_muestras=np.array([len(muestras["masks"])]),
        contornos_exterior=contornos_exterior,
        contornos_r1=contornos_r1,
        contornos_r3=contornos_r3,
        posiciones_agujeros_nominal=np.array(posiciones_agujeros_nominal, dtype=object),
        posicion_r1_nominal=np.array(posicion_r1_nominal, dtype=object),
        posicion_r3_nominal=np.array(posicion_r3_nominal, dtype=object),
    )

    print(f"Shape master guardada correctamente en: {archivo}")
    print(f"Construida a partir de {len(muestras['masks'])} muestras.")


def cargar_shape_master(archivo):
    """
    Carga la referencia master de Shape Matching (máscara consenso +
    lista de contornos por cada tipo + posiciones nominales de
    agujeros/ranuras, provenientes de varias muestras).

    Si el archivo existe pero es de un formato antiguo/incompatible
    (por ejemplo, una master de una sola muestra guardada con una
    versión anterior del script), se detecta explícitamente en vez de
    dejar que falle con un KeyError, y se le indica al operador que
    debe recapturar la master con la tecla 'm'.
    """

    if not os.path.exists(archivo):
        print("No existe shape master todavía.")
        print(f"Captura {NUM_MUESTRAS_MASTER} piezas OK con la tecla 'm' para construirla.")
        return None

    datos = np.load(archivo, allow_pickle=True)

    if "mask_master_promedio" not in datos.files:
        print(f"AVISO: '{archivo}' existe pero es de un formato antiguo/incompatible")
        print("(le falta la clave 'mask_master_promedio'; probablemente es una master")
        print("de una sola muestra guardada con una versión anterior del script).")
        print(f"Bórralo o recaptúralo: presiona 'm' {NUM_MUESTRAS_MASTER} veces para generar uno nuevo.")
        return None

    def cargar_lista_contornos(clave):
        if clave not in datos.files:
            return []
        lista_cruda = datos[clave]
        lista_limpia = []
        for c in lista_cruda:
            if c is None:
                continue
            c_arr = np.asarray(c)
            if c_arr.size == 0:
                continue
            lista_limpia.append(c_arr.astype(np.int32))
        return lista_limpia

    def cargar_posicion(valor):
        # Puede venir como None, o como un array/tupla (x, y).
        if valor is None:
            return None
        arr = np.asarray(valor)
        if arr.size == 0 or arr.item(0) is None:
            return None
        return (float(arr[0]), float(arr[1]))

    posiciones_agujeros_nominal = []
    if "posiciones_agujeros_nominal" in datos.files:
        for v in datos["posiciones_agujeros_nominal"]:
            posiciones_agujeros_nominal.append(cargar_posicion(v))

    shape_master = {
        "mask_master_promedio": datos["mask_master_promedio"],
        "num_muestras": int(datos["num_muestras"][0]) if "num_muestras" in datos.files else None,
        "contornos_exterior": cargar_lista_contornos("contornos_exterior"),
        "contornos_r1": cargar_lista_contornos("contornos_r1"),
        "contornos_r3": cargar_lista_contornos("contornos_r3"),
        "posiciones_agujeros_nominal": posiciones_agujeros_nominal,
        "posicion_r1_nominal": cargar_posicion(datos["posicion_r1_nominal"]) if "posicion_r1_nominal" in datos.files else None,
        "posicion_r3_nominal": cargar_posicion(datos["posicion_r3_nominal"]) if "posicion_r3_nominal" in datos.files else None,
    }

    print(f"Shape master cargada correctamente ({shape_master['num_muestras']} muestras).")
    return shape_master


def registrar_muestra_calibracion(archivo, etiqueta, resultado_shape):
    """
    Agrega una fila al CSV de calibración con los scores de Shape
    Matching del frame actual y una etiqueta manual (OK/NOK) dada por
    el operador. Con varias decenas de piezas buenas y malas
    etiquetadas, este log permite elegir UMBRAL_SHAPE_MASCARA y
    UMBRAL_SHAPE_EXTERIOR con datos reales en vez de a prueba y error.
    """
    import csv
    from datetime import datetime

    os.makedirs(os.path.dirname(archivo), exist_ok=True)
    existe = os.path.exists(archivo)

    with open(archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not existe:
            writer.writerow([
                "timestamp", "etiqueta",
                "score_mascara", "score_exterior",
                "score_r1", "score_r3"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            etiqueta,
            resultado_shape.get("score_mascara"),
            resultado_shape.get("score_exterior"),
            resultado_shape.get("score_r1"),
            resultado_shape.get("score_r3"),
        ])

    print(f"Muestra registrada como '{etiqueta}' en {archivo}")


def calcular_score_shape(referencia, actual):
    """
    Calcula el score de similitud entre dos formas.
    Menor score = mayor similitud.

    Acepta tanto contornos (arrays de puntos) como máscaras binarias
    (imágenes). Antes de llamar a matchShapes valida que ambos objetos
    tengan contenido real, para evitar que una máscara vacía o un
    contorno degenerado tumben el ciclo de inspección.
    """

    if referencia is None or actual is None:
        return None

    try:
        # Validación de contenido: un contorno vacío o una máscara
        # completamente negra no representan una forma comparable.
        if hasattr(referencia, "size") and referencia.size == 0:
            return None
        if hasattr(actual, "size") and actual.size == 0:
            return None

        # Si son imágenes (máscaras), deben tener al menos un pixel activo.
        if referencia.ndim == 2 and cv2.countNonZero(referencia) == 0:
            return None
        if actual.ndim == 2 and cv2.countNonZero(actual) == 0:
            return None

        score = cv2.matchShapes(
            referencia,
            actual,
            cv2.CONTOURS_MATCH_I1,
            0.0
        )

        if score is None or not np.isfinite(score):
            return None

        return score

    except Exception as e:
        # Se captura cualquier excepción (no solo cv2.error) para que un
        # problema de forma/tipo de dato nunca tumbe el ciclo de video
        # en vivo. Se deja registro en consola para poder diagnosticarlo.
        print(f"[shape_matching] Error calculando score: {e}")
        return None


def calcular_score_shape_mejor_de_lista(lista_referencias, actual):
    """
    Compara 'actual' contra cada contorno de la lista de masters y
    devuelve el MEJOR (menor) score encontrado, junto con todos los
    scores individuales para diagnóstico.

    Esto es lo que hace robusta la comparación frente a varias
    muestras: si la pieza actual se parece a CUALQUIERA de las piezas
    buenas capturadas, se considera un match, en vez de exigir que se
    parezca exactamente a una sola referencia arbitraria.
    """
    if not lista_referencias or actual is None:
        return None, []

    scores = []

    for referencia in lista_referencias:
        s = calcular_score_shape(referencia, actual)
        if s is not None:
            scores.append(s)

    if not scores:
        return None, []

    return min(scores), scores


def evaluar_score_shape(score, umbral):
    """
    Evalúa si un score de Shape Matching está dentro del umbral permitido.
    """

    if score is None:
        return "NOK"

    if score <= umbral:
        return "OK"
    else:
        return "NOK"


def evaluar_shape_matching(
    shape_master,
    mask_shape_actual,
    contorno_exterior_actual,
    ranuras_detectadas
):
    """
    Evalúa Shape Matching de:
    - máscara completa de la pieza, contra la máscara consenso
      (promedio de varias muestras buenas)
    - contorno exterior, contra la mejor coincidencia entre todas las
      muestras exteriores guardadas
    - ranura recta, opcional, mismo criterio de mejor coincidencia
    - ranura cuadrada, opcional, mismo criterio
    """

    if shape_master is None:
        return {
            "score_mascara": None,
            "estado_mascara": "SIN_MASTER",
            "score_exterior": None,
            "estado_exterior": "SIN_MASTER",
            "score_r1": None,
            "estado_r1": "SIN_MASTER",
            "score_r3": None,
            "estado_r3": "SIN_MASTER",
            "estado_shape_global": "SIN_MASTER"
        }

    # 1. Comparación de máscara completa contra la máscara consenso
    # (promedio binarizado de todas las muestras). Esta máscara incluye
    # contorno exterior + agujeros + ranuras.
    score_mascara = calcular_score_shape(
        shape_master["mask_master_promedio"],
        mask_shape_actual
    )

    estado_mascara = evaluar_score_shape(
        score_mascara,
        UMBRAL_SHAPE_MASCARA
    )

    # 2. Comparación de contorno exterior: mejor coincidencia entre
    # todas las muestras master guardadas.
    score_exterior, _scores_exterior = calcular_score_shape_mejor_de_lista(
        shape_master["contornos_exterior"],
        contorno_exterior_actual
    )

    estado_exterior = evaluar_score_shape(
        score_exterior,
        UMBRAL_SHAPE_EXTERIOR
    )

    # 3. Comparación opcional de ranura recta
    r1_actual = obtener_primera_ranura_por_tipo(ranuras_detectadas, "recta")

    score_r1, _scores_r1 = calcular_score_shape_mejor_de_lista(
        shape_master["contornos_r1"],
        r1_actual["contorno"] if r1_actual is not None else None
    )

    estado_r1 = evaluar_score_shape(
        score_r1,
        UMBRAL_SHAPE_RANURA_RECTA
    )

    # 4. Comparación opcional de ranura cuadrada
    r3_actual = obtener_primera_ranura_por_tipo(ranuras_detectadas, "cuadrada")

    score_r3, _scores_r3 = calcular_score_shape_mejor_de_lista(
        shape_master["contornos_r3"],
        r3_actual["contorno"] if r3_actual is not None else None
    )

    estado_r3 = evaluar_score_shape(
        score_r3,
        UMBRAL_SHAPE_RANURA_CUADRADA
    )

    condiciones_shape = [
        estado_mascara == "OK",
        estado_exterior == "OK"
    ]

    if USAR_SHAPE_RANURAS:
        condiciones_shape.append(estado_r1 == "OK")
        condiciones_shape.append(estado_r3 == "OK")

    if all(condiciones_shape):
        estado_shape_global = "OK"
    else:
        estado_shape_global = "NOK"

    return {
        "score_mascara": score_mascara,
        "estado_mascara": estado_mascara,
        "score_exterior": score_exterior,
        "estado_exterior": estado_exterior,
        "score_r1": score_r1,
        "estado_r1": estado_r1,
        "score_r3": score_r3,
        "estado_r3": estado_r3,
        "estado_shape_global": estado_shape_global
    }


def detectar_contorno_pieza(binaria, area_minima=10000):
    """
    Detecta el contorno exterior más grande de la pieza.
    """
    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_validos = []

    for c in contornos:
        area = cv2.contourArea(c)

        if area > area_minima:
            contornos_validos.append(c)

    if len(contornos_validos) == 0:
        return None

    return max(contornos_validos, key=cv2.contourArea)


def detectar_agujeros_circulares(holes_mask, px_per_mm, area_minima_px, circularidad_minima):
    """
    Detecta regiones internas oscuras y filtra únicamente las que tienen forma circular.
    """
    contornos, _ = cv2.findContours(
        holes_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    agujeros = []

    for c in contornos:
        area_px = cv2.contourArea(c)

        if area_px < area_minima_px:
            continue

        perimetro = cv2.arcLength(c, True)

        if perimetro == 0:
            continue

        circularidad = calcular_circularidad(c)

        epsilon = 0.02 * perimetro
        approx = cv2.approxPolyDP(c, epsilon, True)
        numero_vertices = len(approx)

        if numero_vertices < 8:
            continue

        if circularidad < circularidad_minima:
            continue

        (x, y), radio_px = cv2.minEnclosingCircle(c)

        if radio_px <= 0:
            continue

        area_circulo_env_px = np.pi * (radio_px ** 2)
        relleno_circular = area_px / area_circulo_env_px

        if relleno_circular < 0.78:
            continue

        diametro_area_mm = obtener_diametro_equivalente_mm(area_px, px_per_mm)
        diametro_circulo_mm = (2 * radio_px) / px_per_mm

        M = cv2.moments(c)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(x), int(y)

        agujeros.append({
            "contorno": c,
            "cx": cx,
            "cy": cy,
            "area_px": area_px,
            "circularidad": circularidad,
            "relleno_circular": relleno_circular,
            "vertices": numero_vertices,
            "diametro_area_mm": diametro_area_mm,
            "diametro_circulo_mm": diametro_circulo_mm
        })

    agujeros = sorted(
        agujeros,
        key=lambda a: a["diametro_area_mm"],
        reverse=True
    )

    return agujeros


def crear_asignacion_agujeros_vacia():
    """Devuelve un diccionario fijo para que C1, C2 y C3 nunca cambien de ID."""
    return {esperado["id"]: None for esperado in AGUJEROS_ESPERADOS}


def asignar_agujeros_para_master(agujeros_detectados):
    """
    Asigna IDs durante la captura inicial de la Shape Master.

    Como todavía no existen posiciones nominales:
    - C1 se identifica por su diámetro cercano a 20 mm.
    - C2 y C3 se identifican espacialmente con una convención estable.

    Esta función se usa solo para crear/recapturar la master. Durante la
    inspección normal se usa asignar_agujeros_por_posicion(), que compara cada
    detección con las posiciones nominales guardadas en la master.
    """
    asignados = crear_asignacion_agujeros_vacia()

    if not agujeros_detectados:
        return asignados

    disponibles = list(agujeros_detectados)

    # C1: candidato cuyo diámetro está más cerca de su nominal (20 mm).
    nominal_c1 = AGUJEROS_ESPERADOS[0]["nominal"]
    idx_c1 = min(
        range(len(disponibles)),
        key=lambda idx: abs(disponibles[idx]["diametro_area_mm"] - nominal_c1)
    )
    c1 = disponibles.pop(idx_c1)
    asignados["C1"] = c1

    # C2/C3: tomar los dos candidatos más parecidos al nominal pequeño.
    nominal_pequeno = AGUJEROS_ESPERADOS[1]["nominal"]
    pequenos = sorted(
        disponibles,
        key=lambda ag: abs(ag["diametro_area_mm"] - nominal_pequeno)
    )[:2]

    if not pequenos:
        return asignados

    if len(pequenos) == 1:
        # Sin posiciones nominales y con un solo círculo pequeño no es posible
        # saber con certeza si corresponde a C2 o C3. La captura de master
        # rechazará esta muestra hasta que estén presentes los tres círculos.
        asignados["C2"] = pequenos[0]
        return asignados

    eje = EJE_ORDEN_C2_C3.lower()
    if eje == "auto":
        dx = abs(pequenos[0]["cx"] - pequenos[1]["cx"])
        dy = abs(pequenos[0]["cy"] - pequenos[1]["cy"])
        eje = "x" if dx >= dy else "y"

    clave = "cx" if eje == "x" else "cy"
    pequenos.sort(
        key=lambda ag: ag[clave],
        reverse=not C2_ORDEN_ASCENDENTE
    )

    asignados["C2"] = pequenos[0]
    asignados["C3"] = pequenos[1]
    return asignados


def asignar_agujeros_por_posicion(
    agujeros_detectados,
    shape_master,
    px_per_mm,
    distancia_max_mm=DISTANCIA_MAX_ASIGNACION_AGUJERO_MM
):
    """
    Relaciona cada círculo detectado con C1, C2 o C3 usando su posición
    nominal de la Shape Master.

    La asignación es uno-a-uno: una misma detección no puede ocupar dos IDs.
    Los pares se procesan por distancia ascendente, por lo que si falta C2,
    el círculo ubicado en C3 se asigna primero a C3 y C2 queda explícitamente
    como ausente. Esto evita el corrimiento por índices que antes convertía
    cualquier círculo faltante en C3.
    """
    if shape_master is None:
        return asignar_agujeros_para_master(agujeros_detectados)

    posiciones_nominal = shape_master.get("posiciones_agujeros_nominal", [])
    if len(posiciones_nominal) < len(AGUJEROS_ESPERADOS):
        return asignar_agujeros_para_master(agujeros_detectados)

    if any(pos is None for pos in posiciones_nominal[:len(AGUJEROS_ESPERADOS)]):
        return asignar_agujeros_para_master(agujeros_detectados)

    asignados = crear_asignacion_agujeros_vacia()
    pares = []

    for idx_esperado, esperado in enumerate(AGUJEROS_ESPERADOS):
        nominal = posiciones_nominal[idx_esperado]

        for idx_detectado, ag in enumerate(agujeros_detectados):
            cx_mm = ag["cx"] / px_per_mm
            cy_mm = ag["cy"] / px_per_mm
            distancia = (
                (cx_mm - nominal[0]) ** 2
                + (cy_mm - nominal[1]) ** 2
            ) ** 0.5

            pares.append((distancia, esperado["id"], idx_detectado))

    pares.sort(key=lambda p: p[0])
    ids_ocupados = set()
    detecciones_ocupadas = set()

    for distancia, id_ag, idx_detectado in pares:
        if distancia > distancia_max_mm:
            continue
        if id_ag in ids_ocupados or idx_detectado in detecciones_ocupadas:
            continue

        asignados[id_ag] = agujeros_detectados[idx_detectado]
        ids_ocupados.add(id_ag)
        detecciones_ocupadas.add(idx_detectado)

    return asignados


def evaluar_agujeros(agujeros_asignados):
    """Evalúa el diámetro de C1, C2 y C3 conservando su identidad espacial."""
    resultados = []

    for esperado in AGUJEROS_ESPERADOS:
        id_ag = esperado["id"]
        nominal = esperado["nominal"]
        tol = esperado["tol"]
        lim_inf = nominal - tol
        lim_sup = nominal + tol
        ag = agujeros_asignados.get(id_ag)

        if ag is not None:
            diametro_mm = ag["diametro_area_mm"]
            estado = evaluar_diametro(diametro_mm, nominal, tol)

            resultados.append({
                "id": id_ag,
                "nominal": nominal,
                "medido": diametro_mm,
                "lim_inf": lim_inf,
                "lim_sup": lim_sup,
                "estado": estado
            })
        else:
            resultados.append({
                "id": id_ag,
                "nominal": nominal,
                "medido": None,
                "lim_inf": lim_inf,
                "lim_sup": lim_sup,
                "estado": "NOK"
            })

    return resultados

def eliminar_circulos_de_mascara(holes_mask, agujeros):
    """
    Elimina de la máscara las regiones que ya fueron clasificadas como círculos.
    """
    mascara_sin_circulos = holes_mask.copy()

    for ag in agujeros:
        cv2.drawContours(
            mascara_sin_circulos,
            [ag["contorno"]],
            -1,
            0,
            thickness=-1
        )

    return mascara_sin_circulos


def detectar_ranuras(mascara_ranuras, px_per_mm, area_minima_px=200):
    """
    Detecta ranuras internas no circulares.
    Usa minAreaRect para obtener largo y ancho.
    """
    contornos, _ = cv2.findContours(
        mascara_ranuras,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    ranuras = []

    for c in contornos:
        area_px = cv2.contourArea(c)

        if area_px < area_minima_px:
            continue

        perimetro = cv2.arcLength(c, True)

        if perimetro == 0:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w_px, h_px), angle = rect

        if w_px <= 0 or h_px <= 0:
            continue

        lado_mayor_mm = max(w_px, h_px) / px_per_mm
        lado_menor_mm = min(w_px, h_px) / px_per_mm

        if lado_menor_mm == 0:
            continue

        relacion_aspecto = lado_mayor_mm / lado_menor_mm
        circularidad = calcular_circularidad(c)

        epsilon = 0.02 * perimetro
        approx = cv2.approxPolyDP(c, epsilon, True)
        vertices = len(approx)

        if relacion_aspecto >= 1.5:
            tipo = "recta"
        else:
            tipo = "cuadrada"

        box = cv2.boxPoints(rect)
        box = np.intp(box)

        ranuras.append({
            "contorno": c,
            "box": box,
            "cx": int(cx),
            "cy": int(cy),
            "lado_mayor_mm": lado_mayor_mm,
            "lado_menor_mm": lado_menor_mm,
            "relacion_aspecto": relacion_aspecto,
            "circularidad": circularidad,
            "vertices": vertices,
            "tipo": tipo,
            "area_px": area_px
        })

    ranuras = sorted(
        ranuras,
        key=lambda r: r["lado_mayor_mm"],
        reverse=True
    )

    return ranuras


def evaluar_ranuras(ranuras_detectadas):
    """
    Evalúa R1 y R3 por cantidad y dimensiones nominales.

    Una categoría solo es conforme si se detecta exactamente la cantidad
    esperada de ranuras de ese tipo y sus dimensiones son correctas.
    Esto impide aceptar una placa con una segunda R1 o R3.
    """
    resultados = []

    for esperado in RANURAS_ESPERADAS:
        coincidencias = [
            r for r in ranuras_detectadas
            if r["tipo"] == esperado["tipo"]
        ]
        cantidad = len(coincidencias)
        cantidad_esperada = esperado["cantidad_esperada"]
        estado_cantidad = "OK" if cantidad == cantidad_esperada else "NOK"

        # Mantiene las mediciones de la primera ranura, como el algoritmo
        # original. Si existen extras, TODAS quedan como NOK por cantidad.
        if coincidencias:
            r = coincidencias[0]
            largo_medido = r["lado_mayor_mm"]
            ancho_medido = r["lado_menor_mm"]

            estado_largo = evaluar_medida(
                largo_medido, esperado["largo_nominal"], esperado["tol"]
            )
            estado_ancho = evaluar_medida(
                ancho_medido, esperado["ancho_nominal"], esperado["tol"]
            )
            dimensiones_ok = estado_largo == "OK" and estado_ancho == "OK"
        else:
            largo_medido = None
            ancho_medido = None
            dimensiones_ok = False

        estado = "OK" if estado_cantidad == "OK" and dimensiones_ok else "NOK"
        resultados.append({
            "id": esperado["id"],
            "tipo": esperado["tipo"],
            "largo_medido": largo_medido,
            "ancho_medido": ancho_medido,
            "cantidad_detectada": cantidad,
            "cantidad_esperada": cantidad_esperada,
            "estado_cantidad": estado_cantidad,
            "estado": estado
        })

    return resultados


def evaluar_posiciones(agujeros_asignados, ranuras_detectadas, shape_master, px_per_mm):
    """
    Valida la posición X,Y de cada feature contra la Shape Master.

    Los agujeros ya llegan asignados por ID mediante
    asignar_agujeros_por_posicion(), por lo que la ausencia de C1, C2 o C3 se
    conserva correctamente y no depende de la cantidad ni del orden de los
    círculos detectados.
    """

    if shape_master is None:
        return {
            "resultados": [],
            "estado_posicion_global": "SIN_MASTER"
        }

    resultados = []

    # --- Agujeros ---
    posiciones_nominal = shape_master.get("posiciones_agujeros_nominal", [])

    for idx, esperado in enumerate(AGUJEROS_ESPERADOS):
        id_ag = esperado["id"]
        nominal = posiciones_nominal[idx] if idx < len(posiciones_nominal) else None

        if nominal is None:
            resultados.append({
                "id": id_ag, "tipo": "agujero",
                "medido": None, "nominal": None, "error_mm": None,
                "estado": "SIN_NOMINAL"
            })
            continue

        ag = agujeros_asignados.get(id_ag)
        if ag is None:
            resultados.append({
                "id": id_ag, "tipo": "agujero",
                "medido": None, "nominal": nominal, "error_mm": None,
                "estado": "NOK"
            })
            continue

        cx_mm = ag["cx"] / px_per_mm
        cy_mm = ag["cy"] / px_per_mm

        error_mm = ((cx_mm - nominal[0]) ** 2 + (cy_mm - nominal[1]) ** 2) ** 0.5
        estado = "OK" if error_mm <= POSICION_AGUJERO_TOL_MM else "NOK"

        resultados.append({
            "id": id_ag, "tipo": "agujero",
            "medido": (cx_mm, cy_mm), "nominal": nominal, "error_mm": error_mm,
            "estado": estado
        })

    # --- Ranuras ---
    for id_ranura, tipo_ranura, nominal in [
        ("R1", "recta", shape_master.get("posicion_r1_nominal")),
        ("R3", "cuadrada", shape_master.get("posicion_r3_nominal")),
    ]:
        if nominal is None:
            resultados.append({
                "id": id_ranura, "tipo": "ranura",
                "medido": None, "nominal": None, "error_mm": None,
                "estado": "SIN_NOMINAL"
            })
            continue

        r_actual = obtener_primera_ranura_por_tipo(ranuras_detectadas, tipo_ranura)

        if r_actual is None:
            resultados.append({
                "id": id_ranura, "tipo": "ranura",
                "medido": None, "nominal": nominal, "error_mm": None,
                "estado": "NOK"
            })
            continue

        cx_mm = r_actual["cx"] / px_per_mm
        cy_mm = r_actual["cy"] / px_per_mm

        error_mm = ((cx_mm - nominal[0]) ** 2 + (cy_mm - nominal[1]) ** 2) ** 0.5
        estado = "OK" if error_mm <= POSICION_RANURA_TOL_MM else "NOK"

        resultados.append({
            "id": id_ranura, "tipo": "ranura",
            "medido": (cx_mm, cy_mm), "nominal": nominal, "error_mm": error_mm,
            "estado": estado
        })

    # SIN_NOMINAL no cuenta como falla; únicamente NOK es una falla real.
    if any(r["estado"] == "NOK" for r in resultados):
        estado_global = "NOK"
    else:
        estado_global = "OK"

    return {
        "resultados": resultados,
        "estado_posicion_global": estado_global
    }

def evaluar_resultado_global(
    agujeros_detectados,
    resultados_agujeros,
    resultados_ranuras,
    resultado_exterior,
    resultado_shape,
    resultado_posicion
):
    """
    Resultado global de la pieza.
    Para ser OK:
    - Contorno exterior OK.
    - Centroide OK.
    - Orientación OK.
    - 3 agujeros OK (tamaño).
    - Exactamente una ranura R1 y una R3 (cantidad y tamaño).
    - Posición de agujeros y ranuras OK, si está activado.
    - Shape Matching OK, si está activado.
    """

    exterior_ok = (
        resultado_exterior is not None
        and resultado_exterior["estado_contorno"] == "OK"
        and resultado_exterior["estado_centroide"] == "OK"
        and resultado_exterior["estado_orientacion"] == "OK"
    )

    agujeros_ok = (
        len(agujeros_detectados) == len(AGUJEROS_ESPERADOS)
        and len(resultados_agujeros) == len(AGUJEROS_ESPERADOS)
        and all(r["estado"] == "OK" for r in resultados_agujeros)
    )

    ranuras_ok = (
        len(resultados_ranuras) == len(RANURAS_ESPERADAS)
        and all(r["estado"] == "OK" for r in resultados_ranuras)
    )

    # Caso especial: todavía no existe una pieza master guardada.
    # No se debe reportar como un NOK más (que se confundiría con una
    # pieza defectuosa); se distingue explícitamente para que el
    # operador sepa que debe calibrar antes de inspeccionar en serio.
    if USAR_SHAPE_MATCHING and resultado_shape is not None:
        if resultado_shape["estado_shape_global"] == "SIN_MASTER":
            return "SIN_MASTER"

    if USAR_VALIDACION_POSICION and resultado_posicion is not None:
        if resultado_posicion["estado_posicion_global"] == "SIN_MASTER":
            return "SIN_MASTER"

    if USAR_SHAPE_MATCHING:
        shape_ok = (
            resultado_shape is not None
            and resultado_shape["estado_shape_global"] == "OK"
        )
    else:
        shape_ok = True

    if USAR_VALIDACION_POSICION:
        posicion_ok = (
            resultado_posicion is not None
            and resultado_posicion["estado_posicion_global"] == "OK"
        )
    else:
        posicion_ok = True

    if exterior_ok and agujeros_ok and ranuras_ok and shape_ok and posicion_ok:
        return "OK"
    else:
        return "NOK"


def crear_buffer_muestras_master():
    return {
        "masks": [],
        "contornos_exterior": [],
        "contornos_r1": [],
        "contornos_r3": [],
        "posiciones_agujeros": [],
        "posiciones_r1": [],
        "posiciones_r3": []
    }


def actualizar_veredicto_estable(buffer_veredictos, resultado_actual):
    """
    Agrega el resultado del frame actual al buffer y determina si ya
    hay consenso (todos los últimos N frames coinciden). Si el buffer
    todavía no está lleno, o los resultados no coinciden entre sí, el
    veredicto se reporta como "EVALUANDO...".

    SIN_MASTER nunca se "estabiliza" como aceptación: se reporta tal
    cual para que sea evidente que falta calibrar.
    """
    buffer_veredictos.append(resultado_actual)

    if len(buffer_veredictos) < buffer_veredictos.maxlen:
        return "EVALUANDO..."

    if all(v == buffer_veredictos[0] for v in buffer_veredictos):
        return buffer_veredictos[0]

    return "EVALUANDO..."


def color_por_estado(estado):
    """
    Devuelve color BGR según estado.
    """
    if estado == "OK":
        return (0, 255, 0)

    if estado in ["SIN_MASTER", "SIN_NOMINAL", "EVALUANDO...", "SIN_PIEZA"]:
        return (0, 255, 255)

    if estado is None:
        return (180, 180, 180)

    return (0, 0, 255)


def escribir_linea(panel, texto, y, color=(255, 255, 255), escala=PANEL_ESCALA, grosor=1):
    """
    Escribe una línea en el panel y regresa la siguiente posición y.
    Evita que el texto se corte por arriba o por abajo.
    """
    alto_panel = panel.shape[0]

    # Evita escribir demasiado arriba o demasiado abajo
    if y < PANEL_MARGEN_SUP:
        y = PANEL_MARGEN_SUP

    if y < alto_panel - PANEL_MARGEN_INF:
        cv2.putText(
            panel,
            texto,
            (PANEL_MARGEN_IZQ, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            color,
            grosor,
            cv2.LINE_AA
        )

    return y + PANEL_LINEA


def crear_panel_resultados(
    resultado_global,
    veredicto_estable,
    resultado_exterior,
    resultado_shape,
    resultado_posicion,
    resultados_agujeros,
    resultados_ranuras,
    agujeros,
    ranuras_detectadas,
    muestras_master_pendientes,
    shape_master
):
    """
    Crea una ventana independiente con todos los datos de inspección.
    """

    panel = np.zeros((PANEL_ALTO, PANEL_ANCHO, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y = PANEL_MARGEN_SUP

    # ==============================
    # ENCABEZADO
    # ==============================

    y = escribir_linea(
        panel,
        "PANEL DE INSPECCION",
        y,
        (255, 255, 255),
        escala=0.62,
        grosor=2
    )

    y += 8

    color_global = color_por_estado(resultado_global)
    color_estable = color_por_estado(veredicto_estable)

    y = escribir_linea(
        panel,
        f"Resultado frame: {resultado_global}",
        y,
        color_global,
        escala=0.55,
        grosor=2
    )

    y = escribir_linea(
        panel,
        f"Veredicto estable: {veredicto_estable}",
        y,
        color_estable,
        escala=0.55,
        grosor=2
    )

    if shape_master is None:
        y = escribir_linea(
            panel,
            "Shape master: NO CARGADA",
            y,
            (0, 255, 255),
            grosor=2
        )
    else:
        num_muestras = shape_master.get("num_muestras", "?")
        y = escribir_linea(
            panel,
            f"Shape master: CARGADA ({num_muestras} muestras)",
            y,
            (0, 255, 0)
        )

    capturas_en_curso = len(muestras_master_pendientes["masks"])

    if capturas_en_curso > 0:
        y = escribir_linea(
            panel,
            f"Captura master en curso: {capturas_en_curso}/{NUM_MUESTRAS_MASTER}",
            y,
            (0, 255, 255)
        )

    y += 10

    # ==============================
    # CONTORNO EXTERIOR
    # ==============================

    y = escribir_linea(panel, "CONTORNO EXTERIOR", y, (255, 255, 0), grosor=2)

    if resultado_exterior is None:
        y = escribir_linea(panel, "P1: No detectado", y, (0, 0, 255))
    else:
        color_contorno = color_por_estado(resultado_exterior["estado_contorno"])
        color_centroide = color_por_estado(resultado_exterior["estado_centroide"])
        color_orientacion = color_por_estado(resultado_exterior["estado_orientacion"])

        y = escribir_linea(
            panel,
            f"P1: {resultado_exterior['lado_mayor_mm']:.2f} x {resultado_exterior['lado_menor_mm']:.2f} mm | {resultado_exterior['estado_contorno']}",
            y,
            color_contorno
        )

        y = escribir_linea(
            panel,
            f"Area: {resultado_exterior['area_contorno_mm2']:.1f} mm2 | Error: {resultado_exterior['error_area_porc']:.2f}%",
            y,
            color_contorno
        )

        y = escribir_linea(
            panel,
            f"Relleno rect: {resultado_exterior['factor_relleno_rect']:.4f} | {resultado_exterior['estado_forma_exterior']}",
            y,
            color_por_estado(resultado_exterior["estado_forma_exterior"])
        )

        y = escribir_linea(
            panel,
            f"Esquinas: {resultado_exterior['relleno_min_esquina']:.3f} | {resultado_exterior['estado_esquinas']}",
            y,
            color_por_estado(resultado_exterior["estado_esquinas"])
        )

        y = escribir_linea(
            panel,
            f"Vertices: {resultado_exterior['vertices_exterior']} | {resultado_exterior['estado_vertices_exterior']}",
            y,
            color_por_estado(resultado_exterior["estado_vertices_exterior"])
        )

        y = escribir_linea(
            panel,
            f"Centroide: X={resultado_exterior['cx_mm']:.2f}, Y={resultado_exterior['cy_mm']:.2f} mm | {resultado_exterior['estado_centroide']}",
            y,
            color_centroide
        )

        y = escribir_linea(
            panel,
            f"Orientacion: {resultado_exterior['angulo']:.2f} grados | {resultado_exterior['estado_orientacion']}",
            y,
            color_orientacion
        )

    y += 10

    # ==============================
    # SHAPE MATCHING
    # ==============================

    y = escribir_linea(panel, "SHAPE MATCHING", y, (255, 255, 0), grosor=2)

    if resultado_shape is None:
        y = escribir_linea(panel, "Shape: sin evaluacion", y, (180, 180, 180))
    else:
        color_shape = color_por_estado(resultado_shape["estado_shape_global"])

        if resultado_shape["score_mascara"] is None:
            y = escribir_linea(panel, "Shape mask: SIN MASTER", y, color_shape)
        else:
            y = escribir_linea(
                panel,
                f"Shape mask: {resultado_shape['score_mascara']:.5f} | {resultado_shape['estado_mascara']}",
                y,
                color_por_estado(resultado_shape["estado_mascara"])
            )

        if resultado_shape["score_exterior"] is None:
            y = escribir_linea(panel, "Shape exterior: SIN MASTER", y, color_shape)
        else:
            y = escribir_linea(
                panel,
                f"Shape exterior: {resultado_shape['score_exterior']:.5f} | {resultado_shape['estado_exterior']}",
                y,
                color_por_estado(resultado_shape["estado_exterior"])
            )

        if USAR_SHAPE_RANURAS:
            if resultado_shape["score_r1"] is None:
                y = escribir_linea(panel, "Shape R1: SIN REF", y, color_shape)
            else:
                y = escribir_linea(
                    panel,
                    f"Shape R1: {resultado_shape['score_r1']:.5f} | {resultado_shape['estado_r1']}",
                    y,
                    color_por_estado(resultado_shape["estado_r1"])
                )

            if resultado_shape["score_r3"] is None:
                y = escribir_linea(panel, "Shape R3: SIN REF", y, color_shape)
            else:
                y = escribir_linea(
                    panel,
                    f"Shape R3: {resultado_shape['score_r3']:.5f} | {resultado_shape['estado_r3']}",
                    y,
                    color_por_estado(resultado_shape["estado_r3"])
                )

        y = escribir_linea(
            panel,
            f"Shape global: {resultado_shape['estado_shape_global']}",
            y,
            color_shape,
            grosor=2
        )

    y += 10

    # ==============================
    # POSICIÓN DE FEATURES
    # ==============================

    y = escribir_linea(panel, "POSICION DE FEATURES", y, (255, 255, 0), grosor=2)

    if resultado_posicion is None:
        y = escribir_linea(panel, "Posicion: sin evaluacion", y, (180, 180, 180))
    else:
        color_pos_global = color_por_estado(resultado_posicion["estado_posicion_global"])

        y = escribir_linea(
            panel,
            f"Estado posicion: {resultado_posicion['estado_posicion_global']}",
            y,
            color_pos_global,
            grosor=2
        )

        for r in resultado_posicion["resultados"]:
            if r["estado"] == "SIN_NOMINAL":
                texto = f"{r['id']}: sin posicion nominal"
                color = (0, 255, 255)

            elif r["medido"] is None:
                texto = f"{r['id']}: AUSENTE | {r['estado']}"
                color = (0, 0, 255)

            else:
                texto = (
                    f"{r['id']}: error {r['error_mm']:.2f} mm "
                    f"(X={r['medido'][0]:.1f}, Y={r['medido'][1]:.1f}) | {r['estado']}"
                )
                color = color_por_estado(r["estado"])

            y = escribir_linea(panel, texto, y, color, escala=0.50)

    y += 10

    # ==============================
    # AGUJEROS
    # ==============================

    y = escribir_linea(
        panel,
        f"AGUJEROS DETECTADOS: {len(agujeros)}",
        y,
        (255, 255, 0),
        grosor=2
    )

    for r in resultados_agujeros:
        if r["medido"] is None:
            texto = f"{r['id']}: AUSENTE | Nom {r['nominal']:.1f} mm | {r['estado']}"
        else:
            texto = (
                f"{r['id']}: {r['medido']:.2f} mm "
                f"| Lim {r['lim_inf']:.1f}-{r['lim_sup']:.1f} | {r['estado']}"
            )

        y = escribir_linea(panel, texto, y, color_por_estado(r["estado"]))

    y += 10

    # ==============================
    # RANURAS
    # ==============================

    y = escribir_linea(
        panel,
        f"RANURAS DETECTADAS: {len(ranuras_detectadas)}",
        y,
        (255, 255, 0),
        grosor=2
    )

    for r in resultados_ranuras:
        cantidad = r["cantidad_detectada"]
        esperada = r["cantidad_esperada"]
        if cantidad == 0:
            texto = f"{r['id']}: AUSENTE (0/{esperada}) | NOK"
        elif cantidad != esperada:
            texto = f"{r['id']}: EXCESO ({cantidad}/{esperada}) | NOK"
        else:
            texto = (
                f"{r['id']}: {r['largo_medido']:.2f} x "
                f"{r['ancho_medido']:.2f} mm | {r['estado']}"
            )

        y = escribir_linea(panel, texto, y, color_por_estado(r["estado"]))

    return panel


def inicializar_base_datos(ruta_bd):
    """
    Crea la base de datos y la tabla de inspecciones si todavía no existen.
    """

    os.makedirs(os.path.dirname(ruta_bd), exist_ok=True)

    conexion = sqlite3.connect(ruta_bd)
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspecciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,
            resultado_frame TEXT,
            veredicto_estable TEXT,
            causa_rechazo TEXT,

            estado_contorno TEXT,
            vertices_exterior INTEGER,
            estado_vertices_exterior TEXT,
            lado_mayor_mm REAL,
            lado_menor_mm REAL,
            area_contorno_mm2 REAL,
            error_area_porc REAL,
            factor_relleno_rect REAL,
            estado_esquinas TEXT,
            relleno_min_esquina REAL,
            angulo_grados REAL,
            estado_orientacion TEXT,
            cx_mm REAL,
            cy_mm REAL,
            estado_centroide TEXT,

            estado_shape_global TEXT,
            score_shape_mascara REAL,
            score_shape_exterior REAL,
            score_shape_r1 REAL,
            estado_shape_r1 TEXT,
            score_shape_r3 REAL,
            estado_shape_r3 TEXT,

            estado_posicion_global TEXT,
            error_pos_c1_mm REAL,
            error_pos_c2_mm REAL,
            error_pos_c3_mm REAL,
            error_pos_r1_mm REAL,
            error_pos_r3_mm REAL,

            c1_diametro_mm REAL,
            c1_estado TEXT,
            c2_diametro_mm REAL,
            c2_estado TEXT,
            c3_diametro_mm REAL,
            c3_estado TEXT,

            r1_largo_mm REAL,
            r1_ancho_mm REAL,
            r1_estado TEXT,
            r3_largo_mm REAL,
            r3_ancho_mm REAL,
            r3_estado TEXT,

            imagen_evidencia TEXT,
            imagen_panel TEXT
        )
    """)

    # Migración compatible: añade columnas nuevas sin borrar la base existente.
    cursor.execute("PRAGMA table_info(inspecciones)")
    columnas_existentes = {fila[1] for fila in cursor.fetchall()}

    columnas_nuevas = {
        "vertices_exterior": "INTEGER",
        "estado_vertices_exterior": "TEXT",
        "score_shape_r1": "REAL",
        "estado_shape_r1": "TEXT",
        "score_shape_r3": "REAL",
        "estado_shape_r3": "TEXT",
        "tiempo_inspeccion_s": "REAL",
        "tiempo_espera_registro_s": "REAL",
        "tiempo_brazo_s": "REAL",
        "tiempo_clasificacion_s": "REAL",
        "tiempo_total_ciclo_s": "REAL",
        "movimiento_ejecutado": "INTEGER DEFAULT 0",
        "estado_movimiento": "TEXT",
        "comando_brazo": "TEXT",
        "motivo_bloqueo_brazo": "TEXT",
        "error_movimiento": "TEXT",
    }

    for nombre, tipo_sql in columnas_nuevas.items():
        if nombre not in columnas_existentes:
            cursor.execute(f"ALTER TABLE inspecciones ADD COLUMN {nombre} {tipo_sql}")

    conexion.commit()
    conexion.close()

    print("Base de datos lista:", ruta_bd)


def buscar_resultado_por_id(lista_resultados, id_buscado):
    """
    Busca un resultado dentro de una lista usando el campo 'id'.
    """
    for r in lista_resultados:
        if r.get("id") == id_buscado:
            return r

    return None


def obtener_error_posicion(resultado_posicion, id_buscado):
    """
    Obtiene el error de posición de un feature específico.
    Ejemplo: C1, C2, C3, R1 o R3.
    """

    if resultado_posicion is None:
        return None

    for r in resultado_posicion.get("resultados", []):
        if r.get("id") == id_buscado:
            return r.get("error_mm")

    return None


def obtener_causa_rechazo(
    resultado_global,
    resultado_exterior,
    resultado_shape,
    resultado_posicion,
    resultados_agujeros,
    resultados_ranuras
):
    """
    Genera un texto corto indicando por qué la pieza fue rechazada.
    Si la pieza es OK, devuelve 'Sin defectos'.
    """

    if resultado_global == "OK":
        return "Sin defectos"

    if resultado_global == "SIN_MASTER":
        return "Falta capturar shape master"

    causas = []

    # Contorno exterior
    if resultado_exterior is None:
        causas.append("No se detecta contorno exterior")

    else:

        if resultado_exterior["estado_lado_mayor"] != "OK":
            causas.append("Largo exterior NOK")

        if resultado_exterior["estado_lado_menor"] != "OK":
            causas.append("Ancho exterior NOK")

        if resultado_exterior["estado_forma_exterior"] != "OK":
            causas.append("Forma exterior NOK")

        if resultado_exterior["estado_esquinas"] != "OK":
            causas.append("Esquinas NOK")

        if resultado_exterior.get("estado_vertices_exterior") != "OK":
            vertices = resultado_exterior.get("vertices_exterior")

        if resultado_exterior.get("estado_vertices_exterior") != "OK":
            causas.append("Vertices NOK")

        if resultado_exterior["estado_orientacion"] != "OK":
            causas.append("Orientacion NOK")

        if resultado_exterior["estado_centroide"] != "OK":
            causas.append("Centroide NOK")

    # Shape Matching
    if resultado_shape is not None:
        if resultado_shape["estado_shape_global"] != "OK":
            causas.append("Shape Matching NOK")

    # Posición de features
    if resultado_posicion is not None:
        if resultado_posicion["estado_posicion_global"] != "OK":

            for r in resultado_posicion.get("resultados", []):
                if r["estado"] == "NOK":
                    causas.append(f"{r['id']} fuera de posicion")

    # Agujeros
    for r in resultados_agujeros:
        if r["estado"] != "OK":
            causas.append(f"{r['id']} {r['estado']}")

    # Ranuras: registra la causa exacta cuando hay ranuras de más o faltantes.
    for r in resultados_ranuras:
        cantidad = r.get("cantidad_detectada", 0)
        esperada = r.get("cantidad_esperada", 1)
        if cantidad > esperada:
            causas.append(f"{r['id']} duplicada: {cantidad} detectadas")
        elif cantidad < esperada:
            causas.append(f"{r['id']} ausente: {cantidad} detectadas")
        elif r["estado"] != "OK":
            causas.append(f"{r['id']} dimensiones NOK")

    if not causas:
        causas.append("Rechazo por criterio global")

    return "; ".join(causas)


def guardar_imagenes_evidencia(salida, panel_resultados, timestamp_archivo):
    """
    Guarda imagen de la pieza y del panel de resultados.
    Se llama únicamente para piezas NOK (ver guardar_inspeccion_sqlite),
    así que no llena el disco con piezas buenas que no necesitan
    revisión visual.
    """

    os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)

    ruta_pieza = os.path.join(CARPETA_EVIDENCIAS, f"pieza_NOK_{timestamp_archivo}.png")
    ruta_panel = os.path.join(CARPETA_EVIDENCIAS, f"panel_NOK_{timestamp_archivo}.png")

    cv2.imwrite(ruta_pieza, salida)
    cv2.imwrite(ruta_panel, panel_resultados)

    return ruta_pieza, ruta_panel


def guardar_inspeccion_sqlite(
    ruta_bd,
    resultado_global,
    veredicto_estable,
    resultado_exterior,
    resultado_shape,
    resultado_posicion,
    resultados_agujeros,
    resultados_ranuras,
    salida=None,
    panel_resultados=None,
    tiempo_inspeccion_s=None,
    tiempo_espera_registro_s=None,
    permitir_movimiento_brazo=True,
    motivo_bloqueo_brazo=""
):
    """
    Guarda una inspección completa en SQLite.

    Solo guarda imágenes de evidencia cuando la pieza es NOK (y
    GUARDAR_IMAGENES_SOLO_NOK está activo) — las piezas OK no
    necesitan revisión visual, y así se evita llenar el disco en
    producción continua.
    """

    timestamp = datetime.now().isoformat(timespec="seconds")

    imagen_evidencia = None
    imagen_panel = None

    debe_guardar_imagen = (
        salida is not None
        and panel_resultados is not None
        and (not GUARDAR_IMAGENES_SOLO_NOK or veredicto_estable == "NOK")
    )

    if debe_guardar_imagen:
        timestamp_archivo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        imagen_evidencia, imagen_panel = guardar_imagenes_evidencia(
            salida,
            panel_resultados,
            timestamp_archivo
        )

    causa_rechazo = obtener_causa_rechazo(
        resultado_global,
        resultado_exterior,
        resultado_shape,
        resultado_posicion,
        resultados_agujeros,
        resultados_ranuras
    )

    # Datos de contorno exterior
    if resultado_exterior is not None:
        vertices_exterior = resultado_exterior["vertices_exterior"]
        estado_vertices_exterior = resultado_exterior["estado_vertices_exterior"]
        estado_contorno = resultado_exterior["estado_contorno"]
        lado_mayor_mm = resultado_exterior["lado_mayor_mm"]
        lado_menor_mm = resultado_exterior["lado_menor_mm"]
        area_contorno_mm2 = resultado_exterior["area_contorno_mm2"]
        error_area_porc = resultado_exterior["error_area_porc"]
        factor_relleno_rect = resultado_exterior["factor_relleno_rect"]
        estado_esquinas = resultado_exterior["estado_esquinas"]
        relleno_min_esquina = resultado_exterior["relleno_min_esquina"]
        angulo_grados = resultado_exterior["angulo"]
        estado_orientacion = resultado_exterior["estado_orientacion"]
        cx_mm = resultado_exterior["cx_mm"]
        cy_mm = resultado_exterior["cy_mm"]
        estado_centroide = resultado_exterior["estado_centroide"]
    else:
        estado_contorno = None
        lado_mayor_mm = None
        lado_menor_mm = None
        area_contorno_mm2 = None
        error_area_porc = None
        factor_relleno_rect = None
        estado_esquinas = None
        relleno_min_esquina = None
        angulo_grados = None
        estado_orientacion = None
        cx_mm = None
        cy_mm = None
        estado_centroide = None

    # Datos de Shape Matching
    if resultado_shape is not None:
        estado_shape_global = resultado_shape["estado_shape_global"]
        score_shape_mascara = resultado_shape["score_mascara"]
        score_shape_exterior = resultado_shape["score_exterior"]
        score_shape_r1 = resultado_shape["score_r1"]
        estado_shape_r1 = resultado_shape["estado_r1"]
        score_shape_r3 = resultado_shape["score_r3"]
        estado_shape_r3 = resultado_shape["estado_r3"]

    else:
        estado_shape_global = None
        score_shape_mascara = None
        score_shape_exterior = None
        score_shape_r1 = None
        estado_shape_r1 = None
        score_shape_r3 = None
        estado_shape_r3 = None

    # Datos de posición
    if resultado_posicion is not None:
        estado_posicion_global = resultado_posicion["estado_posicion_global"]
    else:
        estado_posicion_global = None

    error_pos_c1_mm = obtener_error_posicion(resultado_posicion, "C1")
    error_pos_c2_mm = obtener_error_posicion(resultado_posicion, "C2")
    error_pos_c3_mm = obtener_error_posicion(resultado_posicion, "C3")
    error_pos_r1_mm = obtener_error_posicion(resultado_posicion, "R1")
    error_pos_r3_mm = obtener_error_posicion(resultado_posicion, "R3")

    # Agujeros
    c1 = buscar_resultado_por_id(resultados_agujeros, "C1")
    c2 = buscar_resultado_por_id(resultados_agujeros, "C2")
    c3 = buscar_resultado_por_id(resultados_agujeros, "C3")

    c1_diametro_mm = c1["medido"] if c1 is not None else None
    c1_estado = c1["estado"] if c1 is not None else None

    c2_diametro_mm = c2["medido"] if c2 is not None else None
    c2_estado = c2["estado"] if c2 is not None else None

    c3_diametro_mm = c3["medido"] if c3 is not None else None
    c3_estado = c3["estado"] if c3 is not None else None

    # Ranuras
    r1 = buscar_resultado_por_id(resultados_ranuras, "R1")
    r3 = buscar_resultado_por_id(resultados_ranuras, "R3")

    r1_largo_mm = r1["largo_medido"] if r1 is not None else None
    r1_ancho_mm = r1["ancho_medido"] if r1 is not None else None
    r1_estado = r1["estado"] if r1 is not None else None

    r3_largo_mm = r3["largo_medido"] if r3 is not None else None
    r3_ancho_mm = r3["ancho_medido"] if r3 is not None else None
    r3_estado = r3["estado"] if r3 is not None else None

    estado_movimiento_inicial = (
        "PENDIENTE" if permitir_movimiento_brazo else "BLOQUEADO_DIMENSIONES"
    )
    comando_brazo_inicial = veredicto_estable if permitir_movimiento_brazo else None
    motivo_bloqueo_inicial = (
        None if permitir_movimiento_brazo else motivo_bloqueo_brazo
    )

    os.makedirs(os.path.dirname(ruta_bd), exist_ok=True)
    conexion = sqlite3.connect(ruta_bd)
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO inspecciones (
            timestamp,
            resultado_frame,
            veredicto_estable,
            causa_rechazo,

            estado_contorno,
            vertices_exterior,
            estado_vertices_exterior,
            lado_mayor_mm,
            lado_menor_mm,
            area_contorno_mm2,
            error_area_porc,
            factor_relleno_rect,
            estado_esquinas,
            relleno_min_esquina,
            angulo_grados,
            estado_orientacion,
            cx_mm,
            cy_mm,
            estado_centroide,

            estado_shape_global,
            score_shape_mascara,
            score_shape_exterior,
            score_shape_r1,
            estado_shape_r1,
            score_shape_r3,
            estado_shape_r3,

            estado_posicion_global,
            error_pos_c1_mm,
            error_pos_c2_mm,
            error_pos_c3_mm,
            error_pos_r1_mm,
            error_pos_r3_mm,

            c1_diametro_mm,
            c1_estado,
            c2_diametro_mm,
            c2_estado,
            c3_diametro_mm,
            c3_estado,

            r1_largo_mm,
            r1_ancho_mm,
            r1_estado,
            r3_largo_mm,
            r3_ancho_mm,
            r3_estado,

            imagen_evidencia,
            imagen_panel,

            tiempo_inspeccion_s,
            tiempo_espera_registro_s,
            tiempo_brazo_s,
            tiempo_clasificacion_s,
            tiempo_total_ciclo_s,
            movimiento_ejecutado,
            estado_movimiento,
            comando_brazo,
            motivo_bloqueo_brazo,
            error_movimiento
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        resultado_global,
        veredicto_estable,
        causa_rechazo,

        estado_contorno,
        vertices_exterior,
        estado_vertices_exterior,
        lado_mayor_mm,
        lado_menor_mm,
        area_contorno_mm2,
        error_area_porc,
        factor_relleno_rect,
        estado_esquinas,
        relleno_min_esquina,
        angulo_grados,
        estado_orientacion,
        cx_mm,
        cy_mm,
        estado_centroide,

        estado_shape_global,
        score_shape_mascara,
        score_shape_exterior,
        score_shape_r1,
        estado_shape_r1,
        score_shape_r3,
        estado_shape_r3,

        estado_posicion_global,
        error_pos_c1_mm,
        error_pos_c2_mm,
        error_pos_c3_mm,
        error_pos_r1_mm,
        error_pos_r3_mm,

        c1_diametro_mm,
        c1_estado,
        c2_diametro_mm,
        c2_estado,
        c3_diametro_mm,
        c3_estado,

        r1_largo_mm,
        r1_ancho_mm,
        r1_estado,
        r3_largo_mm,
        r3_ancho_mm,
        r3_estado,

        imagen_evidencia,
        imagen_panel,

        tiempo_inspeccion_s,
        tiempo_espera_registro_s,
        None,  # tiempo_brazo_s: se completa al recibir DONE
        None,  # tiempo_clasificacion_s
        None,  # tiempo_total_ciclo_s
        0,
        estado_movimiento_inicial,
        comando_brazo_inicial,
        motivo_bloqueo_inicial,
        None,
    ))

    conexion.commit()
    id_insertado = cursor.lastrowid
    conexion.close()

    print(f"Inspeccion guardada en SQLite. ID={id_insertado} | Resultado={veredicto_estable}")
    print("Causa:", causa_rechazo)
    return id_insertado


def disparar_accion_externa(veredicto_estable):
    """
    PUNTO DE ENGANCHE para el brazo robótico (o cualquier otra acción
    externa: PLC, señal digital, etc.).

    Se llama UNA vez, justo después de que una inspección se guardó
    correctamente en SQLite mediante la tecla de registro manual — es
    decir, exactamente cuando tú decidiste que hay una pieza lista y
    evaluada, ni un frame antes ni uno después.

    Reemplaza el 'pass' de abajo con tu integración real, por ejemplo:
        - Enviar un byte/comando por puerto serie al controlador del brazo.
        - Escribir en un registro Modbus / activar una salida digital.
        - Llamar a la API del fabricante del brazo.

    'veredicto_estable' ("OK" o "NOK") está disponible por si quieres
    que el brazo haga algo distinto según el resultado (por ejemplo,
    depositar la pieza en charola de buenas vs. charola de rechazo).
    """
    # Ejemplo de cómo distinguir el caso si te sirve:
    # if veredicto_estable == "OK":
    #     mover_brazo_a_charola_ok()
    # else:
    #     mover_brazo_a_charola_nok()
    pass

# ==============================
# MOTOR DE INSPECCIÓN PARA GUI
# ==============================

class MotorInspeccion:
    """
    Encapsula la lógica de inspección para usarla desde una GUI PyQt6.

    Esta clase reemplaza el while True + cv2.imshow + cv2.waitKey del script
    original. OpenCV sigue haciendo el procesamiento, pero PyQt6 se encarga de
    mostrar imágenes y recibir botones.
    """

    def __init__(self, camera_index=CAMERA_INDEX, abrir_camara=True):
        self.camera_index = camera_index
        self.cap = None

        self.umbral = 120
        self.invertir = 0
        self.area_interna_min = 400
        self.circularidad_min = 0.85
        self.area_ranura_min = 200

        self.buffer_veredictos = deque(maxlen=FRAMES_CONSECUTIVOS_REQUERIDOS)
        self.muestras_master_pendientes = crear_buffer_muestras_master()
        self.ultimo_tiempo_registro = 0.0
        self.registro_habilitado_manual = REGISTRO_HABILITADO_MANUAL_DEFAULT
        self.ultimo_resultado = None

        # Medición automática de tiempos de cada pieza.
        # Se usa monotonic() para que cambios en el reloj del sistema no
        # alteren las duraciones medidas.
        self.inicio_pieza_monotonic = None
        self.momento_veredicto_estable_monotonic = None
        self.tiempo_inspeccion_actual_s = None
        self._pieza_presente_anterior = False

        self.cargar_calibracion_y_master()
        inicializar_base_datos(ARCHIVO_BD_INSPECCIONES)

        if abrir_camara:
            self.abrir_camara()

    def cargar_calibracion_y_master(self):
        """Carga homografía, calibración de cámara y shape master."""
        global W_JIG_MM, H_JIG_MM

        if not os.path.exists(ARCHIVO_HOMOGRAFIA):
            raise FileNotFoundError(f"No existe el archivo de homografía: {ARCHIVO_HOMOGRAFIA}")

        datos = np.load(ARCHIVO_HOMOGRAFIA, allow_pickle=True)

        self.H_img_to_mm = datos["H_img_to_mm"]
        self.camera_matrix = datos["camera_matrix"]
        self.dist_coeffs = datos["dist_coeffs"]
        self.new_camera_matrix = datos["new_camera_matrix"]

        W_JIG_MM = cargar_float(datos, "W_JIG_MM", 70.18)
        H_JIG_MM = cargar_float(datos, "H_JIG_MM", 40.10)

        self.H_img_to_rect = crear_matriz_rectificacion(self.H_img_to_mm, PX_PER_MM)
        self.ANCHO_RECT_PX = int(round(W_JIG_MM * PX_PER_MM))
        self.ALTO_RECT_PX = int(round(H_JIG_MM * PX_PER_MM))

        self.shape_master = cargar_shape_master(ARCHIVO_SHAPE_MASTER)

    def abrir_camara(self):
        """Abre la cámara."""
        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara index={self.camera_index}")

    def liberar(self):
        """Libera la cámara."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def set_parametros(self, umbral=None, invertir=None, area_interna_min=None,
                       circularidad_min=None, area_ranura_min=None):
        """Actualiza parámetros de segmentación desde la GUI."""
        if umbral is not None:
            self.umbral = int(umbral)
        if invertir is not None:
            self.invertir = int(invertir)
        if area_interna_min is not None:
            self.area_interna_min = int(area_interna_min)
        if circularidad_min is not None:
            self.circularidad_min = float(circularidad_min)
        if area_ranura_min is not None:
            self.area_ranura_min = int(area_ranura_min)

    def leer_y_procesar(self):
        """Lee un frame de cámara y devuelve el resultado procesado."""
        if self.cap is None or not self.cap.isOpened():
            self.abrir_camara()

        # Copia local para que tanto el analizador estático como el código
        # tengan la garantía de que la cámara existe antes de llamar read().
        cap = self.cap
        if cap is None:
            raise RuntimeError("La cámara no quedó inicializada correctamente")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("No se pudo capturar imagen de la cámara")

        return self.procesar_frame(frame)

    def procesar_frame(self, frame):
        """
        Procesa un frame y devuelve un diccionario con imágenes y resultados.
        No muestra ventanas OpenCV ni lee teclas.
        """
        # Marca monotónica del inicio de ESTE frame. Se define aquí porque
        # procesar_frame() también puede llamarse directamente durante pruebas
        # o validaciones, sin pasar por leer_y_procesar().
        marca_inicio_frame = time.monotonic()

        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame.copy()

        frame_corregido = cv2.undistort(
            frame_bgr,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix
        )

        frame_rect = cv2.warpPerspective(
            frame_corregido,
            self.H_img_to_rect,
            (self.ANCHO_RECT_PX, self.ALTO_RECT_PX)
        )

        gris_rect = cv2.cvtColor(frame_rect, cv2.COLOR_BGR2GRAY)
        gris_suave = cv2.GaussianBlur(gris_rect, (5, 5), 0)

        if self.invertir == 0:
            _, binaria = cv2.threshold(gris_suave, self.umbral, 255, cv2.THRESH_BINARY)
        else:
            _, binaria = cv2.threshold(gris_suave, self.umbral, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
        binaria_limpia = cv2.morphologyEx(binaria_limpia, cv2.MORPH_CLOSE, kernel)

        salida = frame_rect.copy()

        holes_mask = np.zeros_like(binaria_limpia)
        mascara_ranuras = np.zeros_like(binaria_limpia)
        mask_shape_actual = np.zeros_like(binaria_limpia)

        agujeros = []
        agujeros_asignados = crear_asignacion_agujeros_vacia()
        ranuras_detectadas = []
        resultados_agujeros = []
        resultados_ranuras = []
        resultado_global = "SIN_PIEZA"
        resultado_exterior = None
        resultado_shape = None
        resultado_posicion = None
        objeto_plausible = False
        seguridad_brazo = evaluar_seguridad_movimiento_brazo(None)
        contorno_pieza = None

        area_pieza_min = int((70 * 40) * (PX_PER_MM ** 2) * 0.4)
        contorno_pieza = detectar_contorno_pieza(binaria_limpia, area_minima=area_pieza_min)

        if contorno_pieza is not None:
            cv2.drawContours(salida, [contorno_pieza], -1, (0, 255, 0), 2)

            resultado_exterior = evaluar_contorno_exterior(
                contorno_pieza,
                PX_PER_MM,
                binaria_limpia.shape
            )

            objeto_plausible = es_objeto_plausible_como_pieza(resultado_exterior)
            seguridad_brazo = evaluar_seguridad_movimiento_brazo(resultado_exterior)

            if not objeto_plausible:
                cv2.putText(
                    salida,
                    "PIEZA NO DETECTADA",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2
                )

            if objeto_plausible and not seguridad_brazo["permitir_movimiento"]:
                cv2.putText(
                    salida,
                    "BRAZO BLOQUEADO: DIMENSIONES FUERA DE RANGO",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2
                )

            cv2.drawContours(salida, [resultado_exterior["box"]], 0, (255, 0, 0), 2)
            cv2.circle(
                salida,
                (resultado_exterior["cx_px"], resultado_exterior["cy_px"]),
                5,
                (0, 0, 255),
                -1
            )

            mask_pieza = np.zeros_like(binaria_limpia)
            cv2.drawContours(mask_pieza, [contorno_pieza], -1, 255, thickness=-1)

            mask_shape_actual = cv2.bitwise_and(binaria_limpia, mask_pieza)

            holes_mask = cv2.subtract(mask_pieza, binaria_limpia)
            holes_mask = cv2.bitwise_and(holes_mask, mask_pieza)
            holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, kernel)
            holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, kernel)

            agujeros = detectar_agujeros_circulares(
                holes_mask,
                PX_PER_MM,
                self.area_interna_min,
                self.circularidad_min
            )

            mascara_ranuras = eliminar_circulos_de_mascara(holes_mask, agujeros)
            kernel_ranuras = np.ones((3, 3), np.uint8)
            mascara_ranuras = cv2.morphologyEx(mascara_ranuras, cv2.MORPH_OPEN, kernel_ranuras)
            mascara_ranuras = cv2.morphologyEx(mascara_ranuras, cv2.MORPH_CLOSE, kernel_ranuras)

            ranuras_detectadas = detectar_ranuras(
                mascara_ranuras,
                PX_PER_MM,
                area_minima_px=self.area_ranura_min
            )

            agujeros_asignados = asignar_agujeros_por_posicion(
                agujeros,
                self.shape_master,
                PX_PER_MM
            )

            resultados_agujeros = evaluar_agujeros(agujeros_asignados)
            resultados_ranuras = evaluar_ranuras(ranuras_detectadas)

            resultado_shape = evaluar_shape_matching(
                self.shape_master,
                mask_shape_actual,
                contorno_pieza,
                ranuras_detectadas
            )

            resultado_posicion = evaluar_posiciones(
                agujeros_asignados,
                ranuras_detectadas,
                self.shape_master,
                PX_PER_MM
            )

            resultado_global = evaluar_resultado_global(
                agujeros,
                resultados_agujeros,
                resultados_ranuras,
                resultado_exterior,
                resultado_shape,
                resultado_posicion
            )

            # Un objeto ajeno nunca debe aparecer como NOK. Aunque las
            # funciones de medición hayan producido resultados, se fuerza
            # SIN_PIEZA cuando no supera el filtro de identidad geométrica.
            if not objeto_plausible:
                resultado_global = "SIN_PIEZA"
                resultado_shape = None
                resultado_posicion = None
                resultados_agujeros = []
                resultados_ranuras = []
                agujeros = []
                agujeros_asignados = crear_asignacion_agujeros_vacia()
                ranuras_detectadas = []

            # ==========================================================
            # DIBUJO DE AGUJEROS C1, C2 Y C3 SEGÚN SU ESTADO
            # Verde = OK
            # Rojo = NOK
            # ==========================================================

            for id_ag, ag in agujeros_asignados.items():
                if ag is None:
                    continue

                # Resultado dimensional
                res_dim = buscar_resultado_por_id(
                    resultados_agujeros,
                    id_ag
                )

                # Resultado de posición
                res_pos = None
                if resultado_posicion is not None:
                    res_pos = buscar_resultado_por_id(
                        resultado_posicion.get("resultados", []),
                        id_ag
                    )

                estado_dimension = (
                    res_dim is not None
                    and res_dim.get("estado") == "OK"
                )

                # SIN_NOMINAL no se considera falla
                estado_posicion = (
                    res_pos is None
                    or res_pos.get("estado") in ("OK", "SIN_NOMINAL")
                )

                feature_ok = estado_dimension and estado_posicion

                if feature_ok:
                    color_agujero = (0, 200, 0)      # Verde
                    estado_texto = "OK"
                else:
                    color_agujero = (0, 0, 255)      # Rojo
                    estado_texto = "NOK"

                # Contorno del agujero
                cv2.drawContours(
                    salida,
                    [ag["contorno"]],
                    -1,
                    color_agujero,
                    2
                )

                # Centro del agujero
                cv2.circle(
                    salida,
                    (ag["cx"], ag["cy"]),
                    4,
                    color_agujero,
                    -1
                )

                # Texto
                cv2.putText(
                    salida,
                    f"{id_ag} D={ag['diametro_area_mm']:.2f}mm {estado_texto}",
                    (ag["cx"] + 8, ag["cy"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color_agujero,
                    1
                )


            # ==========================================================
            # DIBUJO DE RANURAS R1 Y R3 SEGÚN SU ESTADO
            # Verde = OK
            # Rojo = NOK
            # ==========================================================

            for r in ranuras_detectadas:

                if r["tipo"] == "recta":
                    id_ranura = "R1"
                else:
                    id_ranura = "R3"

                # Resultado dimensional
                res_dim = buscar_resultado_por_id(
                    resultados_ranuras,
                    id_ranura
                )

                # Resultado de posición
                res_pos = None
                if resultado_posicion is not None:
                    res_pos = buscar_resultado_por_id(
                        resultado_posicion.get("resultados", []),
                        id_ranura
                    )

                estado_dimension = (
                    res_dim is not None
                    and res_dim.get("estado") == "OK"
                )

                estado_posicion = (
                    res_pos is None
                    or res_pos.get("estado") in ("OK", "SIN_NOMINAL")
                )

                feature_ok = estado_dimension and estado_posicion

                if feature_ok:
                    color_ranura = (0, 200, 0)       # Verde
                    estado_texto = "OK"
                else:
                    color_ranura = (0, 0, 255)       # Rojo
                    estado_texto = "NOK"

                # Rectángulo de la ranura
                cv2.drawContours(
                    salida,
                    [r["box"]],
                    0,
                    color_ranura,
                    2
                )

                # Centro
                cv2.circle(
                    salida,
                    (r["cx"], r["cy"]),
                    4,
                    color_ranura,
                    -1
                )

                # Texto
                cv2.putText(
                    salida,
                    f"{id_ranura} "
                    f"{r['lado_mayor_mm']:.2f}x{r['lado_menor_mm']:.2f}mm "
                    f"{estado_texto}",
                    (r["cx"] + 8, r["cy"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color_ranura,
                    1
                )


            if resultado_global == "SIN_MASTER":
                # Posición fija segura para el aviso. Evita depender de una
                # variable y_texto que no existe dentro de procesar_frame().
                y_aviso_master = 100
                cv2.putText(
                    salida,
                    "CALIBRAR: coloca la pieza master OK y captura master",
                    (10, y_aviso_master),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )
        else:
            cv2.putText(
                salida,
                "PIEZA NO DETECTADA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------------
        # MEDICIÓN AUTOMÁTICA DEL TIEMPO DE INSPECCIÓN
        # ----------------------------------------------------------
        # El ciclo empieza cuando aparece por primera vez una pieza
        # plausible después de haber tenido el jig vacío. Al entrar una
        # nueva pieza se limpia el buffer para exigir 8 frames nuevos.
        if objeto_plausible:
            if not self._pieza_presente_anterior:
                self.inicio_pieza_monotonic = marca_inicio_frame
                self.momento_veredicto_estable_monotonic = None
                self.tiempo_inspeccion_actual_s = None
                self.buffer_veredictos.clear()
            resultado_para_buffer = resultado_global
        else:
            resultado_para_buffer = "SIN_PIEZA"
            self.inicio_pieza_monotonic = None
            self.momento_veredicto_estable_monotonic = None
            self.tiempo_inspeccion_actual_s = None

        self._pieza_presente_anterior = objeto_plausible

        veredicto_estable = actualizar_veredicto_estable(
            self.buffer_veredictos,
            resultado_para_buffer
        )

        if (
            objeto_plausible
            and veredicto_estable in ("OK", "NOK")
            and self.tiempo_inspeccion_actual_s is None
            and self.inicio_pieza_monotonic is not None
        ):
            self.momento_veredicto_estable_monotonic = time.monotonic()
            self.tiempo_inspeccion_actual_s = (
                self.momento_veredicto_estable_monotonic
                - self.inicio_pieza_monotonic
            )

        panel_resultados = crear_panel_resultados(
            resultado_global,
            veredicto_estable,
            resultado_exterior,
            resultado_shape,
            resultado_posicion,
            resultados_agujeros,
            resultados_ranuras,
            agujeros,
            ranuras_detectadas,
            self.muestras_master_pendientes,
            self.shape_master
        )

        tiempo_actual = time.time()
        cooldown_cumplido = (tiempo_actual - self.ultimo_tiempo_registro) >= COOLDOWN_BOTON_SEG

        pieza_lista_para_registro = (
            objeto_plausible
            and self.registro_habilitado_manual
            and veredicto_estable in ["OK", "NOK"]
            and cooldown_cumplido
        )

        if not self.registro_habilitado_manual:
            texto_registro = "Registro: PAUSADO"
            color_registro = (0, 255, 255)
        elif not objeto_plausible:
            texto_registro = "Registro: PIEZA NO DETECTADA"
            color_registro = (0, 0, 255)
        elif veredicto_estable not in ["OK", "NOK"]:
            texto_registro = f"Registro: esperando veredicto ({veredicto_estable})"
            color_registro = (0, 255, 255)
        elif not cooldown_cumplido:
            texto_registro = "Registro: cooldown"
            color_registro = (0, 255, 255)
        elif not seguridad_brazo["permitir_movimiento"]:
            texto_registro = "Listo para registrar | BRAZO BLOQUEADO POR DIMENSIONES"
            color_registro = (0, 165, 255)
        else:
            texto_registro = "Listo para registrar"
            color_registro = (0, 255, 0)



        data = {
            "salida": salida,
            "panel_resultados": panel_resultados,
            "binaria_limpia": binaria_limpia,
            "holes_mask": holes_mask,
            "mascara_ranuras": mascara_ranuras,
            "resultado_global": resultado_global,
            "veredicto_estable": veredicto_estable,
            "resultado_exterior": resultado_exterior,
            "resultado_shape": resultado_shape,
            "resultado_posicion": resultado_posicion,
            "resultados_agujeros": resultados_agujeros,
            "resultados_ranuras": resultados_ranuras,
            "agujeros": agujeros,
            "agujeros_asignados": agujeros_asignados,
            "ranuras_detectadas": ranuras_detectadas,
            "contorno_pieza": contorno_pieza,
            "mask_shape_actual": mask_shape_actual,
            "objeto_plausible": objeto_plausible,
            "seguridad_brazo": seguridad_brazo,
            "permitir_movimiento_brazo": seguridad_brazo["permitir_movimiento"],
            "tiempo_inspeccion_s": self.tiempo_inspeccion_actual_s,
            "inicio_ciclo_monotonic": self.inicio_pieza_monotonic,
            "momento_veredicto_estable_monotonic": self.momento_veredicto_estable_monotonic,
            "pieza_lista_para_registro": pieza_lista_para_registro,
            "registro_habilitado_manual": self.registro_habilitado_manual,
            "texto_registro": texto_registro,
            "shape_master_cargada": self.shape_master is not None,
            "shape_master_num_muestras": (
                self.shape_master.get("num_muestras") if self.shape_master is not None else None
            ),
            "capturas_master_en_curso": len(self.muestras_master_pendientes["masks"]),
        }

        self.ultimo_resultado = data
        return data

    def registrar_inspeccion_actual(self):
        """Guarda la última inspección y devuelve datos para el brazo/tiempos.

        Devuelve:
            (ok, mensaje, veredicto, permitir_movimiento, motivo_bloqueo,
             id_inspeccion, inicio_ciclo_monotonic, tiempo_inspeccion_s)

        El tiempo de inspección se mide desde la primera detección plausible
        hasta el primer veredicto estable. La espera entre ese veredicto y
        la pulsación de registro también queda guardada por separado.
        """
        data = self.ultimo_resultado
        if data is None:
            return (False, "Aún no hay frame procesado.", None, False, "", None, None, None)

        if not self.registro_habilitado_manual:
            return (False, "Registro pausado.", None, False, "", None, None, None)
        if not data["objeto_plausible"]:
            return (False, "Pieza no detectada.", None, False, "", None, None, None)
        if data["veredicto_estable"] not in ["OK", "NOK"]:
            return (
                False,
                f"Veredicto todavía no estable: {data['veredicto_estable']}",
                None,
                False,
                "",
                None,
                None,
                None,
            )
        if (time.time() - self.ultimo_tiempo_registro) < COOLDOWN_BOTON_SEG:
            return (False, "Cooldown activo. Intenta nuevamente.", None, False, "", None, None, None)

        veredicto = data["veredicto_estable"]
        seguridad_brazo = data.get(
            "seguridad_brazo",
            evaluar_seguridad_movimiento_brazo(data.get("resultado_exterior")),
        )
        permitir_movimiento = bool(
            seguridad_brazo.get("permitir_movimiento", False)
        )
        motivo_bloqueo = seguridad_brazo.get(
            "motivo",
            "Dimensiones fuera del rango seguro.",
        )

        momento_registro = time.monotonic()
        tiempo_inspeccion_s = data.get("tiempo_inspeccion_s")
        momento_estable = data.get("momento_veredicto_estable_monotonic")
        inicio_ciclo = data.get("inicio_ciclo_monotonic")

        if momento_estable is not None:
            tiempo_espera_registro_s = max(0.0, momento_registro - momento_estable)
        else:
            tiempo_espera_registro_s = None

        id_inspeccion = guardar_inspeccion_sqlite(
            ARCHIVO_BD_INSPECCIONES,
            data["resultado_global"],
            veredicto,
            data["resultado_exterior"],
            data["resultado_shape"],
            data["resultado_posicion"],
            data["resultados_agujeros"],
            data["resultados_ranuras"],
            data["salida"],
            data["panel_resultados"],
            tiempo_inspeccion_s=tiempo_inspeccion_s,
            tiempo_espera_registro_s=tiempo_espera_registro_s,
            permitir_movimiento_brazo=permitir_movimiento,
            motivo_bloqueo_brazo=motivo_bloqueo,
        )

        self.ultimo_tiempo_registro = time.time()
        return (
            True,
            f"Inspección registrada: {veredicto}",
            veredicto,
            permitir_movimiento,
            motivo_bloqueo,
            id_inspeccion,
            inicio_ciclo,
            tiempo_inspeccion_s,
        )

    def marcar_movimiento_iniciado(self, id_inspeccion, comando):
        """Marca en SQLite que el brazo comenzó a ejecutar la trayectoria."""
        if id_inspeccion is None:
            return

        conexion = sqlite3.connect(ARCHIVO_BD_INSPECCIONES)
        cursor = conexion.cursor()
        cursor.execute(
            """
            UPDATE inspecciones
            SET movimiento_ejecutado = 1,
                estado_movimiento = 'EN_EJECUCION',
                comando_brazo = ?,
                error_movimiento = NULL
            WHERE id = ?
            """,
            (comando, id_inspeccion),
        )
        conexion.commit()
        conexion.close()

    def finalizar_movimiento_inspeccion(
        self,
        id_inspeccion,
        comando,
        tiempo_brazo_s,
        tiempo_total_ciclo_s,
        tiempo_inspeccion_s=None,
    ):
        """Completa los tiempos de una inspección cuando Arduino responde DONE."""
        if id_inspeccion is None:
            return

        if tiempo_inspeccion_s is not None and tiempo_brazo_s is not None:
            tiempo_clasificacion_s = tiempo_inspeccion_s + tiempo_brazo_s
        else:
            tiempo_clasificacion_s = None

        conexion = sqlite3.connect(ARCHIVO_BD_INSPECCIONES)
        cursor = conexion.cursor()
        cursor.execute(
            """
            UPDATE inspecciones
            SET movimiento_ejecutado = 1,
                estado_movimiento = 'COMPLETADO',
                comando_brazo = ?,
                tiempo_brazo_s = ?,
                tiempo_clasificacion_s = ?,
                tiempo_total_ciclo_s = ?,
                error_movimiento = NULL
            WHERE id = ?
            """,
            (
                comando,
                tiempo_brazo_s,
                tiempo_clasificacion_s,
                tiempo_total_ciclo_s,
                id_inspeccion,
            ),
        )
        conexion.commit()
        conexion.close()

    def registrar_error_movimiento(self, id_inspeccion, mensaje):
        """Guarda el error del robot ligado a la inspección pendiente."""
        if id_inspeccion is None:
            return

        conexion = sqlite3.connect(ARCHIVO_BD_INSPECCIONES)
        cursor = conexion.cursor()
        cursor.execute(
            """
            UPDATE inspecciones
            SET estado_movimiento = 'ERROR',
                error_movimiento = ?
            WHERE id = ?
            """,
            (mensaje, id_inspeccion),
        )
        conexion.commit()
        conexion.close()

    def capturar_master_actual(self):
        """Captura la última pieza plausible como muestra master."""
        data = self.ultimo_resultado
        if data is None:
            return False, "Aún no hay frame procesado."
        if data["contorno_pieza"] is None or not data["objeto_plausible"]:
            return False, "No se detecta pieza plausible para master."

        # Antes de agregar cualquier dato al buffer, se valida que los tres
        # círculos estén presentes y que puedan identificarse como C1/C2/C3.
        # Así una captura incompleta no contamina la Shape Master.
        agujeros_master = asignar_agujeros_para_master(data["agujeros"])
        resultados_master = evaluar_agujeros(agujeros_master)

        faltantes = [
            r["id"] for r in resultados_master
            if r["medido"] is None
        ]
        fuera_medida = [
            r["id"] for r in resultados_master
            if r["medido"] is not None and r["estado"] != "OK"
        ]

        if faltantes:
            return False, f"No se capturó la muestra: faltan {', '.join(faltantes)}."
        if fuera_medida:
            return False, f"No se capturó la muestra: fuera de medida {', '.join(fuera_medida)}."

        r1_actual = obtener_primera_ranura_por_tipo(data["ranuras_detectadas"], "recta")
        r3_actual = obtener_primera_ranura_por_tipo(data["ranuras_detectadas"], "cuadrada")

        posiciones_agujeros_muestra = []
        for esperado in AGUJEROS_ESPERADOS:
            id_ag = esperado["id"]
            agujero_master = agujeros_master.get(id_ag)

            # Esta condición ya debería haber sido descartada por `faltantes`,
            # pero se conserva como defensa explícita y para garantizar el tipo.
            if agujero_master is None:
                return False, f"No se capturó la muestra: falta {id_ag}."

            posiciones_agujeros_muestra.append((
                agujero_master["cx"] / PX_PER_MM,
                agujero_master["cy"] / PX_PER_MM,
            ))

        # Solo después de superar todas las validaciones se agrega la muestra
        # completa al buffer.
        self.muestras_master_pendientes["masks"].append(
            data["mask_shape_actual"].copy()
        )
        self.muestras_master_pendientes["contornos_exterior"].append(
            data["contorno_pieza"].copy()
        )
        self.muestras_master_pendientes["contornos_r1"].append(
            r1_actual["contorno"].copy() if r1_actual is not None else None
        )
        self.muestras_master_pendientes["contornos_r3"].append(
            r3_actual["contorno"].copy() if r3_actual is not None else None
        )
        self.muestras_master_pendientes["posiciones_agujeros"].append(
            posiciones_agujeros_muestra
        )
        self.muestras_master_pendientes["posiciones_r1"].append(
            (r1_actual["cx"] / PX_PER_MM, r1_actual["cy"] / PX_PER_MM)
            if r1_actual is not None else None
        )
        self.muestras_master_pendientes["posiciones_r3"].append(
            (r3_actual["cx"] / PX_PER_MM, r3_actual["cy"] / PX_PER_MM)
            if r3_actual is not None else None
        )

        capturadas = len(self.muestras_master_pendientes["masks"])
        if capturadas >= NUM_MUESTRAS_MASTER:
            guardar_shape_master_multiple(
                ARCHIVO_SHAPE_MASTER,
                self.muestras_master_pendientes
            )
            self.shape_master = cargar_shape_master(ARCHIVO_SHAPE_MASTER)
            inicializar_base_datos(ARCHIVO_BD_INSPECCIONES)
            self.buffer_veredictos.clear()
            self.ultimo_tiempo_registro = 0.0
            self.muestras_master_pendientes = crear_buffer_muestras_master()
            return True, "Nueva Shape Master guardada y cargada."

        faltan = NUM_MUESTRAS_MASTER - capturadas
        return True, (
            f"Muestra master {capturadas}/{NUM_MUESTRAS_MASTER} capturada. "
            f"Faltan {faltan}."
        )

    def reiniciar_master_pendiente(self):
        self.muestras_master_pendientes = crear_buffer_muestras_master()
        return "Captura de muestras master reiniciada."

    def toggle_registro(self):
        self.registro_habilitado_manual = not self.registro_habilitado_manual
        return self.registro_habilitado_manual

    def cargar_indicadores_operacion(self):
        """
        Devuelve indicadores agregados de operación calculados desde SQLite.

        Se consideran todas las inspecciones registradas, no solo las filas
        visibles en la tabla de historial. El tiempo promedio usa únicamente
        ciclos robóticos completados que tengan tiempo_total_ciclo_s válido.
        """
        if not os.path.exists(ARCHIVO_BD_INSPECCIONES):
            return {
                "inspecciones_totales": 0,
                "ok": 0,
                "nok": 0,
                "movimientos_bloqueados": 0,
                "fallos_brazo": 0,
                "tiempo_promedio_ciclo_s": None,
            }

        conexion = sqlite3.connect(ARCHIVO_BD_INSPECCIONES)
        cursor = conexion.cursor()
        try:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS inspecciones_totales,
                    SUM(CASE WHEN veredicto_estable = 'OK' THEN 1 ELSE 0 END) AS total_ok,
                    SUM(CASE WHEN veredicto_estable = 'NOK' THEN 1 ELSE 0 END) AS total_nok,
                    SUM(CASE WHEN estado_movimiento = 'BLOQUEADO_DIMENSIONES' THEN 1 ELSE 0 END)
                        AS movimientos_bloqueados,
                    SUM(CASE WHEN estado_movimiento = 'ERROR' THEN 1 ELSE 0 END)
                        AS fallos_brazo,
                    AVG(
                        CASE
                            WHEN estado_movimiento = 'COMPLETADO'
                                 AND tiempo_total_ciclo_s IS NOT NULL
                            THEN tiempo_total_ciclo_s
                            ELSE NULL
                        END
                    ) AS tiempo_promedio_ciclo_s
                FROM inspecciones
                """
            )
            fila = cursor.fetchone()
        finally:
            conexion.close()

        if fila is None:
            return {
                "inspecciones_totales": 0,
                "ok": 0,
                "nok": 0,
                "movimientos_bloqueados": 0,
                "fallos_brazo": 0,
                "tiempo_promedio_ciclo_s": None,
            }

        return {
            "inspecciones_totales": int(fila[0] or 0),
            "ok": int(fila[1] or 0),
            "nok": int(fila[2] or 0),
            "movimientos_bloqueados": int(fila[3] or 0),
            "fallos_brazo": int(fila[4] or 0),
            "tiempo_promedio_ciclo_s": (
                float(fila[5]) if fila[5] is not None else None
            ),
        }


    def cargar_historial(self, limite=50):
        """Devuelve las últimas inspecciones para mostrarlas en la GUI."""
        if not os.path.exists(ARCHIVO_BD_INSPECCIONES):
            return []

        conexion = sqlite3.connect(ARCHIVO_BD_INSPECCIONES)
        cursor = conexion.cursor()
        cursor.execute("""
            SELECT
                id,
                timestamp,
                veredicto_estable,
                causa_rechazo,
                estado_contorno,
                c1_estado,
                c2_estado,
                c3_estado,
                r1_estado,
                r3_estado
            FROM inspecciones
            ORDER BY id DESC
            LIMIT ?
        """, (limite,))
        filas = cursor.fetchall()
        conexion.close()
        return filas