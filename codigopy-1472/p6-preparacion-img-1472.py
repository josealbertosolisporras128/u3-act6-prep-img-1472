# Práctica 6: Preparación de imágenes para IA - José Solís (NC: 1472)
import cv2
import numpy as np
import os
import csv

# 1. Configuración inicial
AUTOR = "José Solís"
NC = "1472"
carpeta_original = f"img-original-{NC}"
carpeta_preparadas = f"img-preparadas-{NC}"
carpeta_resultados = f"resultados-img-{NC}"

ANCHO_DESTINO = 224
ALTO_DESTINO = 224

# Crear carpetas de salida si no existen
os.makedirs(carpeta_preparadas, exist_ok=True)
os.makedirs(carpeta_resultados, exist_ok=True)

categorias = ["bosque", "desierto"]
reporte = []
imagenes_procesadas = 0
imagenes_aprobadas = 0

print(f"{AUTOR.lower()} nc {NC}")
print(f"🚀 Iniciando preparación de imágenes ({AUTOR})...\n")

# 2. Recorrer cada categoría y sus archivos
for categoria in categorias:
    ruta_origen = os.path.join(carpeta_original, categoria)
    ruta_destino = os.path.join(carpeta_preparadas, categoria)
    
    os.makedirs(ruta_destino, exist_ok=True)
    
    if not os.path.exists(ruta_origen):
        print(f"⚠️ La carpeta {ruta_origen} no existe. Saltando...\n")
        continue
        
    for nombre_archivo in sorted(os.listdir(ruta_origen)):
        if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            continue
            
        ruta_imagen = os.path.join(ruta_origen, nombre_archivo)
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        formato_orig = os.path.splitext(nombre_archivo)[1].upper().replace(".", "")
        
        print(f"📸 Procesando [{categoria}]: {nombre_archivo}")
        
        # 3. Leer la imagen e identificar tamaño y formato
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"   ❌ Error: No se pudo leer la imagen.\n")
            reporte.append([AUTOR, categoria, nombre_archivo, formato_orig, "JPG", "ERROR", "-", "-", "-", "-", "-", "-", "REVISAR"])
            continue
            
        imagenes_procesadas += 1
        alto_orig, ancho_orig, canales = imagen.shape
        print(f"   📄 Formato original: {formato_orig} -> JPG | Dimensiones: {ancho_orig} x {alto_orig}")
        
        # 4. Cambiar el tamaño a dimensiones uniformes (224x224)
        imagen_redimensionada = cv2.resize(imagen, (ANCHO_DESTINO, ALTO_DESTINO), interpolation=cv2.INTER_AREA)
        print(f"   📐 Dimensiones nuevas: {ANCHO_DESTINO} x {ALTO_DESTINO}")
        
        # Convertir y guardar en formato común (.jpg)
        nombre_jpg_final = f"{nombre_sin_ext}.jpg"
        ruta_jpg = os.path.join(ruta_destino, nombre_jpg_final)
        cv2.imwrite(ruta_jpg, imagen_redimensionada)
        
        # 5. Normalizar los valores de los píxeles (0-255 a 0.0-1.0)
        imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0
        val_min = float(np.min(imagen_normalizada))
        val_max = float(np.max(imagen_normalizada))
        print(f"   🔢 Rango normalizado: {val_min:.3f} a {val_max:.3f}")
        
        # Guardar matriz normalizada (.npy)
        ruta_npy = os.path.join(ruta_destino, f"{nombre_sin_ext}.npy")
        np.save(ruta_npy, imagen_normalizada)
        
        # 6. Evaluar que cumpla con los criterios establecidos
        dimensiones_correctas = (imagen_redimensionada.shape[0] == ALTO_DESTINO and 
                                 imagen_redimensionada.shape[1] == ANCHO_DESTINO)
        rango_correcto = (val_min >= 0.0 and val_max <= 1.0)
        archivos_guardados = os.path.isfile(ruta_jpg) and os.path.isfile(ruta_npy)
        
        if dimensiones_correctas and rango_correcto and archivos_guardados:
            estado = "APROBADA"
            imagenes_aprobadas += 1
            print(f"   ✅ Evaluación: APROBADA\n")
        else:
            estado = "REVISAR"
            print(f"   ⚠️ Evaluación: REVISAR\n")
            
        # 7. Guardar datos en el reporte
        reporte.append([
            AUTOR, categoria, nombre_archivo, formato_orig, "JPG", "OK", ancho_orig, alto_orig,
            ANCHO_DESTINO, ALTO_DESTINO, round(val_min, 3), round(val_max, 3), estado
        ])

# 8. Generar reporte en CSV
ruta_reporte = os.path.join(carpeta_resultados, f"reporte_preparacion_{NC}.csv")
with open(ruta_reporte, "w", newline="", encoding="utf-8") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow([
        "Autor", "Categoría", "Archivo original", "Formato orig.", "Formato final", "Lectura", 
        "Ancho original", "Alto original", "Ancho preparado", "Alto preparado", 
        "Valor mínimo norm.", "Valor máximo norm.", "Evaluación"
    ])
    escritor.writerows(reporte)

# 9. Resumen final en consola
print("=" * 45)
print(f"🎉 RESUMEN DE LA PRÁCTICA - {AUTOR.upper()}")
print("=" * 45)
print(f"Imágenes procesadas: {imagenes_procesadas}")
print(f"Imágenes aprobadas:  {imagenes_aprobadas}")
print(f"Imágenes a revisar:  {imagenes_procesadas - imagenes_aprobadas}")
print(f"\n📁 Reporte guardado en: {ruta_reporte}")
print("✅ PRÁCTICA TERMINADA EXITOSAMENTE")