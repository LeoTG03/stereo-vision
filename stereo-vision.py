import numpy as np
import cv2 as cv
import argparse as arg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_image(path):
    """
    Lee una imagen de un archivo.

    Args:
        path (str): Ruta del archivo de imagen.

    Returns:
        numpy.ndarray: Imagen leída.
    """
    img = cv.imread(path)
    if img is None:
        print(f"Error: No se puede leer la imagen {path}")
    return img

def read_parser():
    """
    Lee los argumentos de la línea de comandos.

    Returns:
        argparse.Namespace: Argumentos de la línea de comandos parseados.
    """
    parser = arg.ArgumentParser(description="Program for the reconstruction of a 3D space using stereo vision")
    parser.add_argument("--l_img", 
                        dest="left_image", 
                        type=str, 
                        help="Selecciona una imagen con el ángulo izquierdo de la escena")
    parser.add_argument("--r_img", 
                        dest="right_image", 
                        type=str, 
                        help="Selecciona una imagen con el ángulo derecho de la escena")
    args = parser.parse_args()
    
    return args

def draw_circle(image, center):
    """
    Dibuja un círculo pequeño alrededor del píxel seleccionado.

    Args:
        image (numpy.ndarray): Imagen en la que se dibujará el círculo.
        center (tuple): Coordenadas del centro del círculo.

    Returns:
        None
    """
    cv.circle(image, center, radius=5, color=(0, 255, 0), thickness=1)

def calcular_coordenadas(uL, vL, uR, vR):
    """
    Calcula las coordenadas 3D de un punto dado en las imágenes estéreo.

    Args:
        uL (int): Coordenada x del punto en la imagen izquierda.
        vL (int): Coordenada y del punto en la imagen izquierda.
        uR (int): Coordenada x del punto en la imagen derecha.
        vR (int): Coordenada y del punto en la imagen derecha.

    Returns:
        tuple: Coordenadas X, Y, Z en milímetros.
    """
    baseline = 94.926  # Distancia entre las cámaras en mm
    fx = 648.52  # Longitud focal en píxeles
    fy = 648.52
    Cx = 635.709
    Cy = 370.88

    ucl = uL - Cx
    vcl = vL - Cy

    ucr = uR - Cx
    vcr = vR - Cy

    'ucl = 670 - Cx'
    'vcl = 442 - Cy'

    'ucr = 588 - Cx'
    'vcr = 442 - Cy'

    # Calcular la disparidad
    d = ucl - ucr

    if d == 0:
        print("Disparidad es cero, no se puede calcular la profundidad.")
        return None

    # Calcular la profundidad (Z)
    Z = (baseline * fx) / d

    # Calcular las coordenadas X y Y
    X = (ucl) * Z / fx
    Y = (vcl) * Z / fy

    print(X, Y, Z)

    return X, Y, Z

def reconstruccion_3D(pixel_imgl, pixel_imgr):
    """
    Genera una reconstrucción 3D a partir de los puntos seleccionados en las imágenes izquierda y derecha.

    Args:
        pixel_imgl (list): Lista de puntos seleccionados en la imagen izquierda.
        pixel_imgr (list): Lista de puntos seleccionados en la imagen derecha.

    Returns:
        None
    """
    puntos_3D = []

    for (uL, vL), (uR, vR) in zip(pixel_imgl, pixel_imgr):
        coords = calcular_coordenadas(uL, vL, uR, vR)
        if coords:
            puntos_3D.append(coords)

    puntos_3D = np.array(puntos_3D)

    # Visualizar la reconstrucción 3D
    fig = plt.figure()
    figura3d = fig.add_subplot(111, projection='3d')
    figura3d.scatter(puntos_3D[:, 0], puntos_3D[:, 1], puntos_3D[:, 2], c='b', marker='o')
    figura3d.set_xlabel('X')
    figura3d.set_ylabel('Y')
    figura3d.set_zlabel('Z')
    plt.show()

def select_points(left_image, right_image, num_points):
    """
    Permite seleccionar puntos en ambas imágenes de manera intercalada.

    Args:
        left_image (numpy.ndarray): Imagen izquierda.
        right_image (numpy.ndarray): Imagen derecha.
        num_points (int): Número de puntos a seleccionar.

    Returns:
        list, list: Listas de puntos seleccionados en la imagen izquierda y derecha.
    """
    points_left = []
    points_right = []

    def Mouse_interaction(event, x, y, flags, param):
        nonlocal points_left, points_right
        if event == cv.EVENT_LBUTTONDOWN:
            if param == 'left' and len(points_left) < num_points:
                points_left.append((x, y))
                draw_circle(left_image, (x, y))
                cv.imshow('Imagen Izquierda', left_image)
            elif param == 'right' and len(points_right) < num_points:
                points_right.append((x, y))
                draw_circle(right_image, (x, y))
                cv.imshow('Imagen Derecha', right_image)

    cv.imshow('Imagen Izquierda', left_image)
    cv.imshow('Imagen Derecha', right_image)
    cv.setMouseCallback('Imagen Izquierda', Mouse_interaction, param='left')
    cv.setMouseCallback('Imagen Derecha', Mouse_interaction, param='right')

    while len(points_left) < num_points or len(points_right) < num_points:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    return points_left, points_right

def main():
    """
    Función principal que ejecuta el programa de reconstrucción 3D usando visión estéreo.

    Returns:
        None
    """
    global left_image, right_image, original_left_image, original_right_image, args

    args = read_parser()
    left_image = read_image(args.left_image)
    if left_image is None:           
        return

    right_image = read_image(args.right_image)
    if right_image is None:
        return

    # Hacer copias de las imágenes originales para reiniciar después
    original_left_image = left_image.copy()
    original_right_image = right_image.copy()

    # Seleccionar puntos en ambas imágenes
    print("Seleccione 30 puntos en las imágenes. Presione 'q' para terminar antes de tiempo.")
    pixel_imgl, pixel_imgr = select_points(left_image, right_image, 30)

    if len(pixel_imgl) == len(pixel_imgr):
        reconstruccion_3D(pixel_imgl, pixel_imgr)
    else:
        print("Número de puntos seleccionados no coincide entre las dos imágenes.")

if __name__ == "__main__":
    main()
