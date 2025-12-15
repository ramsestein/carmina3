#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Anonimización - Paso 1: Extracción de entidades con Regex

Este script busca entidades en los archivos de test usando regex basándose en
entities_reference.json, excluyendo las entradas "formatos" y "referencias".
"""

import json
import re
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Intentar importar rapidfuzz para fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("ADVERTENCIA: rapidfuzz no está instalado. Instálalo con: pip install rapidfuzz", flush=True)
    print("  El fuzzy matching no estará disponible.", flush=True)

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def load_entities_reference(entities_file: str = "pipeline_anon/entities_reference.json") -> Dict[str, List[str]]:
    """
    Carga el archivo de referencia de entidades y extrae todas las entidades
    excepto las que están en claves llamadas "formatos" y "referencias".
    
    Args:
        entities_file: Ruta al archivo JSON de referencia
        
    Returns:
        Dict con las entidades organizadas por categoría
    """
    print(f"Cargando entidades de referencia desde: {entities_file}", flush=True)
    
    if not os.path.exists(entities_file):
        raise FileNotFoundError(f"No se encontró el archivo: {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities_dict = {}
    entities_list = []
    
    # Recorrer todas las categorías principales
    if 'entities' in data:
        for category, subcategories in data['entities'].items():
            if isinstance(subcategories, dict):
                category_entities = []
                for subcategory, values in subcategories.items():
                    # Excluir "formatos" y "referencias"
                    if subcategory.lower() not in ['formatos', 'referencias']:
                        if isinstance(values, list):
                            category_entities.extend(values)
                            entities_list.extend(values)
                        elif isinstance(values, str):
                            category_entities.append(values)
                            entities_list.append(values)
                
                if category_entities:
                    entities_dict[category] = category_entities
    
    # Eliminar duplicados manteniendo el orden
    entities_list = list(dict.fromkeys(entities_list))
    
    print(f"  Categorías encontradas: {len(entities_dict)}", flush=True)
    print(f"  Total entidades únicas: {len(entities_list)}", flush=True)
    
    return entities_dict, entities_list

def find_entities_in_text(text: str, entities: List[str], fuzzy_threshold: float = 85.0, use_fuzzy: bool = True) -> List[str]:
    """
    Busca todas las entidades en el texto usando regex (búsqueda exacta) y fuzzy matching.
    Versión optimizada que busca todas las entidades de una vez.
    
    Args:
        text: Texto donde buscar
        entities: Lista de entidades a buscar
        fuzzy_threshold: Umbral de similitud para fuzzy matching (0-100, default: 85)
        use_fuzzy: Si True, usa fuzzy matching además de regex exacto
        
    Returns:
        Lista de entidades encontradas (puede contener duplicados)
    """
    found_entities = []
    
    # Crear un conjunto de entidades únicas y ordenadas por longitud (más largas primero)
    # para evitar coincidencias parciales
    unique_entities = [e for e in set(entities) if e and e.strip()]
    sorted_entities = sorted(unique_entities, key=len, reverse=True)
    
    # Crear un diccionario para rastrear qué entidades ya fueron encontradas
    # y en qué posiciones (para evitar solapamientos)
    found_positions = {}  # (start, end) -> entity
    
    # Dividir el texto en palabras para fuzzy matching más eficiente
    # Extraer todas las posibles subcadenas de palabras que podrían ser entidades
    words = re.findall(r'\b\w+\b', text.lower())
    text_lower = text.lower()
    
    # Buscar cada entidad
    for entity in sorted_entities:
        entity_lower = entity.lower().strip()
        entity_len = len(entity)
        
        # 1. Búsqueda exacta con regex (más rápida)
        escaped_entity = re.escape(entity)
        pattern_str = r'(?<![^\s\n\r\t.,;:!?()\[\]{}\'"])' + escaped_entity + r'(?![^\s\n\r\t.,;:!?()\[\]{}\'"])'
        pattern = re.compile(pattern_str, re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start_pos = match.start()
            end_pos = match.end()
            
            # Verificar solapamientos
            is_overlapping = False
            for (pos_start, pos_end), found_entity in found_positions.items():
                if start_pos >= pos_start and end_pos <= pos_end:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                found_entities.append(entity)
                found_positions[(start_pos, end_pos)] = entity
        
        # 2. Fuzzy matching si está disponible y habilitado
        if use_fuzzy and RAPIDFUZZ_AVAILABLE and entity_len >= 3:  # Solo para entidades de 3+ caracteres
            # Buscar palabras en el texto que sean similares a la entidad
            # Extraer todas las palabras del texto que tengan longitud similar
            word_pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
            for match in word_pattern.finditer(text):
                word = match.group()
                word_lower = word.lower()
                
                # Solo considerar palabras de longitud similar (±2 caracteres)
                if abs(len(word) - entity_len) > 2:
                    continue
                
                # Calcular similitud usando rapidfuzz
                similarity = fuzz.ratio(entity_lower, word_lower)
                
                if similarity >= fuzzy_threshold:
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Verificar solapamientos
                    is_overlapping = False
                    for (pos_start, pos_end), found_entity in found_positions.items():
                        if start_pos >= pos_start and end_pos <= pos_end:
                            is_overlapping = True
                            break
                    
                    if not is_overlapping:
                        found_entities.append(entity)
                        found_positions[(start_pos, end_pos)] = entity
                        # Continuar buscando más ocurrencias de la misma entidad
    
    return found_entities

def process_text_file(file_path: str, entities: List[str], fuzzy_threshold: float = 85.0, use_fuzzy: bool = True) -> Dict:
    """
    Procesa un archivo de texto y encuentra todas las entidades.
    
    Args:
        file_path: Ruta al archivo de texto
        entities: Lista de entidades a buscar
        
    Returns:
        Dict con filename, entities y entity_count
    """
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return {
            "filename": filename,
            "entities": [],
            "entity_count": 0,
            "error": str(e)
        }
    
    # Buscar entidades
    found_entities = find_entities_in_text(text, entities, fuzzy_threshold=fuzzy_threshold, use_fuzzy=use_fuzzy)
    
    return {
        "filename": filename,
        "entities": found_entities,
        "entity_count": len(found_entities)
    }

def process_file_wrapper(args):
    """
    Wrapper para procesar un archivo (necesario para multiprocessing).
    
    Args:
        args: Tupla (file_path, entities, fuzzy_threshold, use_fuzzy)
        
    Returns:
        Dict con los resultados
    """
    file_path, entities, fuzzy_threshold, use_fuzzy = args
    return process_text_file(file_path, entities, fuzzy_threshold=fuzzy_threshold, use_fuzzy=use_fuzzy)

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
    
    parser = argparse.ArgumentParser(description='Extrae entidades usando regex')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Número máximo de archivos a procesar (para pruebas)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Número de workers paralelos (por defecto: número de CPUs)')
    args = parser.parse_args()
    
    # Iniciar medición de tiempo total
    start_time_total = time.time()
    
    print("="*60, flush=True)
    print("PIPELINE DE ANONIMIZACIÓN - PASO 1: EXTRACCIÓN CON REGEX", flush=True)
    print("="*60, flush=True)
    if use_fuzzy and RAPIDFUZZ_AVAILABLE:
        print(f"Fuzzy matching: ACTIVADO (umbral: {args.fuzzy_threshold}%)", flush=True)
    elif use_fuzzy and not RAPIDFUZZ_AVAILABLE:
        print("Fuzzy matching: DESACTIVADO (rapidfuzz no disponible)", flush=True)
    else:
        print("Fuzzy matching: DESACTIVADO (--no-fuzzy)", flush=True)
    
    # Rutas
    entities_file = "pipeline_anon/entities_reference.json"
    test_dir = "docs/test"
    output_file = "pipeline_anon/entities_anon.jsonl"
    
    # Cargar entidades de referencia
    print("\n1. Cargando entidades de referencia...", flush=True)
    start_time = time.time()
    try:
        entities_dict, entities_list = load_entities_reference(entities_file)
        load_time = time.time() - start_time
        print(f"  Tiempo de carga: {format_time(load_time)}", flush=True)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar las entidades: {e}", flush=True)
        return 1
    
    # Obtener lista de archivos .txt en docs/test
    print(f"\n2. Buscando archivos .txt en {test_dir}...", flush=True)
    if not os.path.exists(test_dir):
        print(f"ERROR: No se encontró el directorio {test_dir}", flush=True)
        return 1
    
    txt_files = list(Path(test_dir).glob("*.txt"))
    print(f"  Archivos encontrados: {len(txt_files)}", flush=True)
    
    if len(txt_files) == 0:
        print("ERROR: No se encontraron archivos .txt", flush=True)
        return 1
    
    # Limitar archivos si se especifica
    if args.max_files and args.max_files > 0:
        txt_files = txt_files[:args.max_files]
        print(f"  Limitando procesamiento a {len(txt_files)} archivos (modo prueba)", flush=True)
    
    # Procesar cada archivo en paralelo
    print(f"\n3. Procesando archivos y buscando entidades (paralelo)...", flush=True)
    print(f"  Total archivos a procesar: {len(txt_files)}", flush=True)
    print(f"  Total entidades a buscar: {len(entities_list)}", flush=True)
    
    # Determinar número de workers
    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    print(f"  Usando {num_workers} workers paralelos", flush=True)
    
    start_time = time.time()
    results = []
    total_entities_found = 0
    processed_count = 0
    
    # Preparar argumentos para el pool
    use_fuzzy = not args.no_fuzzy
    file_args = [(str(txt_file), entities_list, args.fuzzy_threshold, use_fuzzy) for txt_file in txt_files]
    
    # Procesar en paralelo
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        future_to_file = {executor.submit(process_file_wrapper, args): args[0] 
                          for args in file_args}
        
        # Procesar resultados conforme se completan
        for future in as_completed(future_to_file):
            processed_count += 1
            try:
                result = future.result()
                results.append(result)
                total_entities_found += result["entity_count"]
                
                # Mostrar progreso cada 10 archivos o al inicio
                if processed_count % 10 == 0 or processed_count == 1:
                    print(f"  Procesados {processed_count}/{len(txt_files)} archivos... "
                          f"(entidades encontradas hasta ahora: {total_entities_found})", flush=True)
            except Exception as e:
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                print(f"  ERROR procesando {filename}: {e}", flush=True)
                results.append({
                    "filename": filename,
                    "entities": [],
                    "entity_count": 0,
                    "error": str(e)
                })
    
    # Ordenar resultados por nombre de archivo para mantener consistencia
    results.sort(key=lambda x: x["filename"])
    
    processing_time = time.time() - start_time
    print(f"  Procesamiento completado: {len(results)} archivos", flush=True)
    print(f"  Total entidades encontradas: {total_entities_found}", flush=True)
    print(f"  Tiempo de procesamiento: {format_time(processing_time)}", flush=True)
    if len(results) > 0:
        avg_time_per_file = processing_time / len(results)
        print(f"  Tiempo promedio por archivo: {format_time(avg_time_per_file)}", flush=True)
    
    # Guardar resultados en JSONL
    print(f"\n4. Guardando resultados en {output_file}...", flush=True)
    start_time = time.time()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    save_time = time.time() - start_time
    print(f"  Resultados guardados: {len(results)} entradas", flush=True)
    print(f"  Tiempo de guardado: {format_time(save_time)}", flush=True)
    
    # Calcular tiempo total
    total_time = time.time() - start_time_total
    
    # Estadísticas
    print(f"\n5. Estadísticas:", flush=True)
    print(f"  Archivos procesados: {len(results)}", flush=True)
    print(f"  Archivos con entidades: {sum(1 for r in results if r['entity_count'] > 0)}", flush=True)
    print(f"  Archivos sin entidades: {sum(1 for r in results if r['entity_count'] == 0)}", flush=True)
    print(f"  Total entidades encontradas: {total_entities_found}", flush=True)
    if len(results) > 0:
        print(f"  Promedio entidades por archivo: {total_entities_found / len(results):.2f}", flush=True)
    
    # Top 10 archivos con más entidades
    sorted_results = sorted(results, key=lambda x: x['entity_count'], reverse=True)
    print(f"\n  Top 10 archivos con más entidades:", flush=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"    {i}. {result['filename']}: {result['entity_count']} entidades", flush=True)
    
    # Resumen final con tiempos
    print("\n" + "="*60, flush=True)
    print("RESUMEN FINAL", flush=True)
    print("="*60, flush=True)
    print(f"Archivos procesados: {len(results)}", flush=True)
    print(f"Entidades encontradas: {total_entities_found}", flush=True)
    print(f"Workers utilizados: {num_workers}", flush=True)
    print(f"\nTiempos:", flush=True)
    print(f"  Carga de entidades: {format_time(load_time)}", flush=True)
    print(f"  Procesamiento: {format_time(processing_time)}", flush=True)
    print(f"  Guardado: {format_time(save_time)}", flush=True)
    print(f"  TOTAL: {format_time(total_time)}", flush=True)
    if len(results) > 0:
        throughput = len(results) / total_time
        print(f"\nRendimiento:", flush=True)
        print(f"  Archivos por segundo: {throughput:.2f}", flush=True)
        print(f"  Entidades por segundo: {total_entities_found / total_time:.2f}", flush=True)
    print("="*60, flush=True)
    print("PASO 1 COMPLETADO", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

