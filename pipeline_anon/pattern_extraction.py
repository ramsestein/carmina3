#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Anonimización - Paso 2: Extracción de patrones con Regex

Este script busca patrones similares a los formatos definidos en entities_reference.json
usando regex genéricos basados en combinaciones de números y letras.
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

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def load_formatos_reference(entities_file: str = "pipeline_anon/entities_reference.json") -> Dict[str, List[str]]:
    """
    Carga solo las categorías "formatos" del archivo de referencia de entidades.
    
    Args:
        entities_file: Ruta al archivo JSON de referencia
        
    Returns:
        Dict con las categorías y sus formatos
    """
    print(f"Cargando formatos de referencia desde: {entities_file}", flush=True)
    
    if not os.path.exists(entities_file):
        raise FileNotFoundError(f"No se encontró el archivo: {entities_file}")
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatos_dict = {}
    
    # Recorrer todas las categorías principales y extraer solo "formatos"
    if 'entities' in data:
        for category, subcategories in data['entities'].items():
            if isinstance(subcategories, dict):
                if 'formatos' in subcategories:
                    formatos_list = subcategories['formatos']
                    if isinstance(formatos_list, list) and formatos_list:
                        formatos_dict[category] = formatos_list
    
    print(f"  Categorías con formatos encontradas: {len(formatos_dict)}", flush=True)
    total_formatos = sum(len(f) for f in formatos_dict.values())
    print(f"  Total formatos de ejemplo: {total_formatos}", flush=True)
    
    return formatos_dict

def generate_patterns_from_formatos(formatos_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Genera patrones regex genéricos basados en los formatos de ejemplo.
    
    Args:
        formatos_dict: Diccionario con categorías y sus formatos de ejemplo
        
    Returns:
        Dict con categorías y sus patrones regex (patrón, descripción)
    """
    patterns_dict = {}
    
    for category, formatos in formatos_dict.items():
        patterns = []
        
        if category == "FECHAS":
            # Patrones de fechas
            patterns.extend([
                (r'\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}', 'Fecha formato DD/MM/YYYY o DD-MM-YYYY (años 19XX o 20XX)'),
                (r'(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}', 'Fecha formato YYYY/MM/DD o YYYY-MM-DD (años 19XX o 20XX)'),
                (r'\d{1,2}\.\d{1,2}\.(?:19|20)\d{2}', 'Fecha formato DD.MM.YYYY (años 19XX o 20XX)'),
                (r'(?:19|20)\d{2}', 'Año de 4 dígitos (19XX o 20XX)'),
            ])
        
        elif category == "NOMBRES_PERSONAS":
            # Patrones de títulos profesionales (texto, no números/letras)
            # Estos son más difíciles de generalizar, pero podemos buscar abreviaciones comunes
            patterns.extend([
                (r'\b(?:Dr|Dra|D|Dña|Dn|Dna|Sr|Sra|Srta|Prof|Enf|Aux|Lic|Ldo|Lda|R[1-5]|MIR|EIR|PIR)\b\.?', 
                 'Títulos profesionales abreviados'),
            ])
        
        elif category == "NUMEROS_IDENTIFICACION":
            # Patrones de números de identificación
            patterns.extend([
                # DNI español: 8 dígitos + letra
                (r'\b\d{8}[A-Z]\b', 'DNI español (8 dígitos + letra)'),
                # DNI con X/Y/Z al inicio
                (r'\b[XYZ]\d{7}[A-Z]\b', 'DNI con letra inicial'),
                # Teléfonos: 9 dígitos
                (r'\b\d{9}\b', 'Teléfono de 9 dígitos'),
                # Teléfonos con prefijo +34
                (r'\+34\s*\d{1,3}\s*\d{3}\s*\d{2}\s*\d{2}', 'Teléfono con prefijo +34'),
                # Teléfonos con guiones o puntos
                (r'\d{2,3}[-.\s]\d{3}[-.\s]\d{2}[-.\s]\d{2}', 'Teléfono con separadores'),
                # Códigos postales: 5 dígitos
                (r'\b\d{5}\b', 'Código postal de 5 dígitos'),
                # NSS: formato 12 12345678 01
                (r'\d{2}\s*\d{8}\s*\d{2}', 'Número de Seguridad Social'),
                # NSS con guiones
                (r'\d{2}[-]\d{8}[-]\d{2}', 'Número de Seguridad Social con guiones'),
            ])
        
        if patterns:
            patterns_dict[category] = patterns
    
    print(f"  Patrones generados para {len(patterns_dict)} categorías", flush=True)
    total_patterns = sum(len(p) for p in patterns_dict.values())
    print(f"  Total patrones regex: {total_patterns}", flush=True)
    
    return patterns_dict

def find_patterns_in_text(text: str, patterns: List[Tuple[str, str]]) -> List[str]:
    """
    Busca patrones regex en el texto.
    TODAS las entidades deben tener espacio o puntuación antes y después.
    
    Args:
        text: Texto donde buscar
        patterns: Lista de tuplas (patrón_regex, descripción)
        
    Returns:
        Lista de entidades encontradas
    """
    found_entities = []
    found_positions = set()
    
    # Patrón para límites: espacio, inicio/fin de línea, o puntuación
    # (?<![^\s\n\r\t.,;:!?()\[\]{}\'"]) -> lookbehind negativo: no debe haber un no-espacio/puntuación antes
    # (?![^\s\n\r\t.,;:!?()\[\]{}\'"])   -> lookahead negativo: no debe haber un no-espacio/puntuación después
    boundary_lookbehind = r'(?<![^\s\n\r\t.,;:!?()\[\]{}\'"])'
    boundary_lookahead = r'(?![^\s\n\r\t.,;:!?()\[\]{}\'"])'
    
    for pattern_str, description in patterns:
        try:
            # Agregar límites antes y después del patrón
            # Nota: algunos patrones ya tienen \b (word boundary), pero necesitamos ser más estrictos
            # con espacios y puntuación
            pattern_with_boundaries = boundary_lookbehind + pattern_str + boundary_lookahead
            pattern = re.compile(pattern_with_boundaries, re.IGNORECASE)
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group(0)
                
                # Verificar si esta posición ya fue encontrada
                is_overlapping = False
                for pos_start, pos_end in found_positions:
                    if start_pos >= pos_start and end_pos <= pos_end:
                        is_overlapping = True
                        break
                
                if not is_overlapping and matched_text.strip():
                    found_entities.append(matched_text)
                    found_positions.add((start_pos, end_pos))
        except re.error as e:
            # Si hay un error en el patrón regex, continuar con el siguiente
            continue
    
    return found_entities

def process_text_file_patterns(file_path: str, patterns_dict: Dict[str, List[Tuple[str, str]]]) -> Dict:
    """
    Procesa un archivo de texto y encuentra todas las entidades usando patrones.
    
    Args:
        file_path: Ruta al archivo de texto
        patterns_dict: Diccionario con categorías y sus patrones
        
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
    
    # Buscar patrones de todas las categorías
    found_entities = []
    for category, patterns in patterns_dict.items():
        category_entities = find_patterns_in_text(text, patterns)
        found_entities.extend(category_entities)
    
    return {
        "filename": filename,
        "entities": found_entities,
        "entity_count": len(found_entities)
    }

def process_file_wrapper_patterns(args):
    """
    Wrapper para procesar un archivo con patrones (necesario para multiprocessing).
    
    Args:
        args: Tupla (file_path, patterns_dict)
        
    Returns:
        Dict con los resultados
    """
    file_path, patterns_dict = args
    return process_text_file_patterns(file_path, patterns_dict)

def load_existing_results(jsonl_file: str) -> Dict[str, Dict]:
    """
    Carga los resultados existentes del JSONL.
    
    Args:
        jsonl_file: Ruta al archivo JSONL
        
    Returns:
        Dict con filename como clave y el dict completo como valor
    """
    results = {}
    if os.path.exists(jsonl_file):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        filename = entry.get('filename')
                        if filename:
                            results[filename] = entry
        except Exception as e:
            print(f"  ADVERTENCIA: Error al cargar resultados existentes: {e}", flush=True)
    return results

def merge_results(existing: Dict[str, Dict], new: Dict) -> Dict:
    """
    Fusiona los resultados nuevos con los existentes.
    
    Args:
        existing: Resultado existente para un archivo
        new: Resultado nuevo para el mismo archivo
        
    Returns:
        Dict fusionado
    """
    if not existing:
        return new
    
    # Combinar entidades y eliminar duplicados manteniendo el orden
    existing_entities = existing.get('entities', [])
    new_entities = new.get('entities', [])
    
    # Combinar y eliminar duplicados
    combined_entities = list(dict.fromkeys(existing_entities + new_entities))
    
    return {
        "filename": new['filename'],
        "entities": combined_entities,
        "entity_count": len(combined_entities)
    }

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
    
    parser = argparse.ArgumentParser(description='Extrae entidades usando patrones regex')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Número máximo de archivos a procesar (para pruebas)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Número de workers paralelos (por defecto: número de CPUs)')
    args = parser.parse_args()
    
    # Iniciar medición de tiempo total
    start_time_total = time.time()
    
    print("="*60, flush=True)
    print("PIPELINE DE ANONIMIZACIÓN - PASO 2: EXTRACCIÓN CON PATRONES", flush=True)
    print("="*60, flush=True)
    
    # Rutas
    entities_file = "pipeline_anon/entities_reference.json"
    test_dir = "docs/test"
    output_file = "pipeline_anon/entities_anon.jsonl"
    
    # Cargar formatos de referencia
    print("\n1. Cargando formatos de referencia...", flush=True)
    start_time = time.time()
    try:
        formatos_dict = load_formatos_reference(entities_file)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los formatos: {e}", flush=True)
        return 1
    
    # Generar patrones regex
    print("\n2. Generando patrones regex...", flush=True)
    patterns_dict = generate_patterns_from_formatos(formatos_dict)
    load_time = time.time() - start_time
    print(f"  Tiempo de carga y generación: {format_time(load_time)}", flush=True)
    
    # Cargar resultados existentes
    print(f"\n3. Cargando resultados existentes desde {output_file}...", flush=True)
    existing_results = load_existing_results(output_file)
    print(f"  Archivos existentes: {len(existing_results)}", flush=True)
    
    # Obtener lista de archivos .txt en docs/test
    print(f"\n4. Buscando archivos .txt en {test_dir}...", flush=True)
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
    print(f"\n5. Procesando archivos y buscando patrones (paralelo)...", flush=True)
    print(f"  Total archivos a procesar: {len(txt_files)}", flush=True)
    
    # Determinar número de workers
    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    print(f"  Usando {num_workers} workers paralelos", flush=True)
    
    start_time = time.time()
    results = []
    total_entities_found = 0
    processed_count = 0
    
    # Preparar argumentos para el pool
    file_args = [(str(txt_file), patterns_dict) for txt_file in txt_files]
    
    # Procesar en paralelo
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        future_to_file = {executor.submit(process_file_wrapper_patterns, args): args[0] 
                          for args in file_args}
        
        # Procesar resultados conforme se completan
        for future in as_completed(future_to_file):
            processed_count += 1
            try:
                new_result = future.result()
                filename = new_result['filename']
                
                # Fusionar con resultados existentes si los hay
                existing = existing_results.get(filename, {})
                merged_result = merge_results(existing, new_result)
                results.append(merged_result)
                total_entities_found += merged_result["entity_count"]
                
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
    print(f"\n6. Guardando resultados en {output_file}...", flush=True)
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
    print(f"\n7. Estadísticas:", flush=True)
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
    print(f"  Carga y generación de patrones: {format_time(load_time)}", flush=True)
    print(f"  Procesamiento: {format_time(processing_time)}", flush=True)
    print(f"  Guardado: {format_time(save_time)}", flush=True)
    print(f"  TOTAL: {format_time(total_time)}", flush=True)
    if len(results) > 0:
        throughput = len(results) / total_time
        print(f"\nRendimiento:", flush=True)
        print(f"  Archivos por segundo: {throughput:.2f}", flush=True)
        print(f"  Entidades por segundo: {total_entities_found / total_time:.2f}", flush=True)
    print("="*60, flush=True)
    print("PASO 2 COMPLETADO", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

