#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Anonimización - Paso 3: Ocultación de entidades

Este script reemplaza todas las entidades detectadas en entities_anon.jsonl
por "XXXX" en los archivos de texto originales.
"""

import json
import re
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def load_entities_from_jsonl(jsonl_file: str) -> Dict[str, List[str]]:
    """
    Carga las entidades detectadas desde el archivo JSONL.
    
    Args:
        jsonl_file: Ruta al archivo JSONL con las entidades
        
    Returns:
        Dict con filename como clave y lista de entidades como valor
    """
    print(f"Cargando entidades desde: {jsonl_file}", flush=True)
    
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"No se encontró el archivo: {jsonl_file}")
    
    entities_by_file = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    filename = entry.get('filename')
                    entities = entry.get('entities', [])
                    if filename and entities:
                        entities_by_file[filename] = entities
                except json.JSONDecodeError as e:
                    print(f"  ADVERTENCIA: Error parseando línea: {e}", flush=True)
                    continue
    
    print(f"  Archivos con entidades: {len(entities_by_file)}", flush=True)
    total_entities = sum(len(e) for e in entities_by_file.values())
    print(f"  Total entidades cargadas: {total_entities}", flush=True)
    
    return entities_by_file

def ocult_entities_in_text(text: str, entities: List[str]) -> str:
    """
    Reemplaza todas las entidades en el texto por "XXXX".
    
    Args:
        text: Texto original
        entities: Lista de entidades a ocultar
        
    Returns:
        Texto con las entidades reemplazadas por "XXXX"
    """
    if not entities:
        return text
    
    # Crear un conjunto de entidades únicas y ordenadas por longitud (más largas primero)
    # para evitar reemplazar partes de entidades más largas
    unique_entities = [e for e in set(entities) if e and e.strip()]
    sorted_entities = sorted(unique_entities, key=len, reverse=True)
    
    # Crear un conjunto para rastrear qué partes del texto ya fueron reemplazadas
    replaced_positions = set()
    result_text = text
    offset = 0
    
    # Crear lista de reemplazos con sus posiciones
    replacements = []
    
    for entity in sorted_entities:
        if not entity or not entity.strip():
            continue
        
        # Escapar caracteres especiales de regex
        escaped_entity = re.escape(entity)
        
        # Para entidades de 3 caracteres o menos, requerir espacios antes y después
        if len(entity) <= 3:
            pattern_str = r'(?<![^\s\n\r\t.,;:!?()\[\]{}\'"])' + escaped_entity + r'(?![^\s\n\r\t.,;:!?()\[\]{}\'"])'
        else:
            pattern_str = escaped_entity
        
        # Buscar todas las ocurrencias
        pattern = re.compile(pattern_str, re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start_pos = match.start()
            end_pos = match.end()
            
            # Verificar si esta posición ya fue marcada para reemplazo
            is_overlapping = False
            for (pos_start, pos_end) in replaced_positions:
                if start_pos >= pos_start and end_pos <= pos_end:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                replacements.append((start_pos, end_pos, entity))
                replaced_positions.add((start_pos, end_pos))
    
    # Ordenar reemplazos por posición (de mayor a menor para poder reemplazar sin afectar índices)
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    # Aplicar reemplazos desde el final hacia el principio
    for start_pos, end_pos, entity in replacements:
        result_text = result_text[:start_pos] + "XXXX" + result_text[end_pos:]
    
    return result_text

def process_file_ocultation(file_path: str, entities: List[str], output_dir: str) -> Dict:
    """
    Procesa un archivo de texto y oculta las entidades.
    
    Args:
        file_path: Ruta al archivo de texto original
        entities: Lista de entidades a ocultar
        output_dir: Directorio de salida
        
    Returns:
        Dict con información del procesamiento
    """
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
    except Exception as e:
        return {
            "filename": filename,
            "success": False,
            "error": str(e),
            "entities_replaced": 0,
            "original_length": 0,
            "new_length": 0
        }
    
    # Ocultar entidades
    oculted_text = ocult_entities_in_text(original_text, entities)
    
    # Contar cuántas entidades fueron reemplazadas
    # Contar ocurrencias de "XXXX" que no estaban en el texto original
    original_xxxx_count = original_text.count("XXXX")
    new_xxxx_count = oculted_text.count("XXXX")
    entities_replaced = new_xxxx_count - original_xxxx_count
    
    # Guardar archivo ocultado
    output_path = os.path.join(output_dir, filename)
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(oculted_text)
    except Exception as e:
        return {
            "filename": filename,
            "success": False,
            "error": f"Error al guardar: {e}",
            "entities_replaced": entities_replaced,
            "original_length": len(original_text),
            "new_length": len(oculted_text)
        }
    
    return {
        "filename": filename,
        "success": True,
        "entities_replaced": entities_replaced,
        "original_length": len(original_text),
        "new_length": len(oculted_text)
    }

def process_file_wrapper_ocult(args):
    """
    Wrapper para procesar un archivo con ocultación (necesario para multiprocessing).
    
    Args:
        args: Tupla (file_path, entities, output_dir)
        
    Returns:
        Dict con los resultados
    """
    file_path, entities, output_dir = args
    return process_file_ocultation(file_path, entities, output_dir)

def format_time(seconds: float) -> str:
    """
    Formatea el tiempo en segundos a un formato legible.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        String formateado (ej: "2m 30.5s" o "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.2f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.2f}s"

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Oculta entidades en textos')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Número máximo de archivos a procesar (para pruebas)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Número de workers paralelos (por defecto: número de CPUs)')
    args = parser.parse_args()
    
    # Iniciar medición de tiempo total
    start_time_total = time.time()
    
    print("="*60, flush=True)
    print("PIPELINE DE ANONIMIZACIÓN - PASO 3: OCULTACIÓN DE ENTIDADES", flush=True)
    print("="*60, flush=True)
    
    # Rutas
    entities_file = "pipeline_anon/entities_anon.jsonl"
    input_dir = "docs/test"
    output_dir = "docs/test_ocult"
    
    # Cargar entidades detectadas
    print("\n1. Cargando entidades detectadas...", flush=True)
    start_time = time.time()
    try:
        entities_by_file = load_entities_from_jsonl(entities_file)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar las entidades: {e}", flush=True)
        return 1
    load_time = time.time() - start_time
    print(f"  Tiempo de carga: {format_time(load_time)}", flush=True)
    
    # Obtener lista de archivos .txt en docs/test
    print(f"\n2. Buscando archivos .txt en {input_dir}...", flush=True)
    if not os.path.exists(input_dir):
        print(f"ERROR: No se encontró el directorio {input_dir}", flush=True)
        return 1
    
    txt_files = list(Path(input_dir).glob("*.txt"))
    print(f"  Archivos encontrados: {len(txt_files)}", flush=True)
    
    if len(txt_files) == 0:
        print("ERROR: No se encontraron archivos .txt", flush=True)
        return 1
    
    # Filtrar solo archivos que tienen entidades detectadas
    files_to_process = []
    for txt_file in txt_files:
        filename = txt_file.name
        if filename in entities_by_file:
            files_to_process.append(txt_file)
    
    print(f"  Archivos con entidades detectadas: {len(files_to_process)}", flush=True)
    
    if len(files_to_process) == 0:
        print("ADVERTENCIA: No hay archivos con entidades detectadas", flush=True)
        return 0
    
    # Limitar archivos si se especifica
    if args.max_files and args.max_files > 0:
        files_to_process = files_to_process[:args.max_files]
        print(f"  Limitando procesamiento a {len(files_to_process)} archivos (modo prueba)", flush=True)
    
    # Procesar cada archivo en paralelo
    print(f"\n3. Procesando archivos y ocultando entidades (paralelo)...", flush=True)
    print(f"  Total archivos a procesar: {len(files_to_process)}", flush=True)
    
    # Determinar número de workers
    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    print(f"  Usando {num_workers} workers paralelos", flush=True)
    
    start_time = time.time()
    results = []
    total_entities_replaced = 0
    processed_count = 0
    
    # Preparar argumentos para el pool
    file_args = [
        (str(txt_file), entities_by_file[txt_file.name], output_dir) 
        for txt_file in files_to_process
    ]
    
    # Procesar en paralelo
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        future_to_file = {executor.submit(process_file_wrapper_ocult, args): args[0] 
                          for args in file_args}
        
        # Procesar resultados conforme se completan
        for future in as_completed(future_to_file):
            processed_count += 1
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    total_entities_replaced += result.get('entities_replaced', 0)
                
                # Mostrar progreso cada 10 archivos o al inicio
                if processed_count % 10 == 0 or processed_count == 1:
                    print(f"  Procesados {processed_count}/{len(files_to_process)} archivos... "
                          f"(entidades reemplazadas hasta ahora: {total_entities_replaced})", flush=True)
            except Exception as e:
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                print(f"  ERROR procesando {filename}: {e}", flush=True)
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                    "entities_replaced": 0,
                    "original_length": 0,
                    "new_length": 0
                })
    
    # Ordenar resultados por nombre de archivo
    results.sort(key=lambda x: x["filename"])
    
    processing_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"  Procesamiento completado: {len(results)} archivos", flush=True)
    print(f"  Exitosos: {successful}", flush=True)
    print(f"  Fallidos: {failed}", flush=True)
    print(f"  Total entidades reemplazadas: {total_entities_replaced}", flush=True)
    print(f"  Tiempo de procesamiento: {format_time(processing_time)}", flush=True)
    if len(results) > 0:
        avg_time_per_file = processing_time / len(results)
        print(f"  Tiempo promedio por archivo: {format_time(avg_time_per_file)}", flush=True)
    
    # Calcular tiempo total
    total_time = time.time() - start_time_total
    
    # Estadísticas
    print(f"\n4. Estadísticas:", flush=True)
    print(f"  Archivos procesados: {len(results)}", flush=True)
    print(f"  Archivos exitosos: {successful}", flush=True)
    print(f"  Archivos fallidos: {failed}", flush=True)
    print(f"  Total entidades reemplazadas: {total_entities_replaced}", flush=True)
    if successful > 0:
        avg_replacements = total_entities_replaced / successful
        print(f"  Promedio entidades reemplazadas por archivo: {avg_replacements:.2f}", flush=True)
    
    # Mostrar archivos fallidos si los hay
    if failed > 0:
        failed_files = [r for r in results if not r['success']]
        print(f"\n  Archivos fallidos:", flush=True)
        for result in failed_files[:10]:  # Mostrar máximo 10
            print(f"    - {result['filename']}: {result.get('error', 'Error desconocido')}", flush=True)
        if len(failed_files) > 10:
            print(f"    ... y {len(failed_files) - 10} más", flush=True)
    
    # Top 10 archivos con más entidades reemplazadas
    successful_results = [r for r in results if r['success']]
    sorted_results = sorted(successful_results, key=lambda x: x.get('entities_replaced', 0), reverse=True)
    if sorted_results:
        print(f"\n  Top 10 archivos con más entidades reemplazadas:", flush=True)
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"    {i}. {result['filename']}: {result.get('entities_replaced', 0)} entidades", flush=True)
    
    # Resumen final con tiempos
    print("\n" + "="*60, flush=True)
    print("RESUMEN FINAL", flush=True)
    print("="*60, flush=True)
    print(f"Archivos procesados: {len(results)}", flush=True)
    print(f"Archivos exitosos: {successful}", flush=True)
    print(f"Entidades reemplazadas: {total_entities_replaced}", flush=True)
    print(f"Workers utilizados: {num_workers}", flush=True)
    print(f"Directorio de salida: {output_dir}", flush=True)
    print(f"\nTiempos:", flush=True)
    print(f"  Carga de entidades: {format_time(load_time)}", flush=True)
    print(f"  Procesamiento: {format_time(processing_time)}", flush=True)
    print(f"  TOTAL: {format_time(total_time)}", flush=True)
    if len(results) > 0:
        throughput = len(results) / total_time
        print(f"\nRendimiento:", flush=True)
        print(f"  Archivos por segundo: {throughput:.2f}", flush=True)
        print(f"  Entidades reemplazadas por segundo: {total_entities_replaced / total_time:.2f}", flush=True)
    print("="*60, flush=True)
    print("PASO 3 COMPLETADO", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

