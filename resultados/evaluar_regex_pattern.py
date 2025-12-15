#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de las métricas de regex_extraction + pattern_extraction.

Este script replica la lógica de regex_extraction.py y pattern_extraction.py
directamente sobre 50 documentos aleatorios para evaluar las métricas.
"""

import json
import re
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Intentar importar rapidfuzz para fuzzy matching
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def load_entities_reference(entities_file: str = "pipeline_anon/entities_reference.json") -> Tuple[Dict, List[str]]:
    """
    Carga el archivo de referencia de entidades y extrae todas las entidades
    excepto las que están en claves llamadas "formatos" y "referencias".
    """
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
    
    return entities_dict, entities_list

def load_formatos_reference(entities_file: str = "pipeline_anon/entities_reference.json") -> Dict[str, List[str]]:
    """
    Carga solo las categorías "formatos" del archivo de referencia de entidades.
    """
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
    
    return formatos_dict

def generate_patterns_from_formatos(formatos_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Genera patrones regex genéricos basados en los formatos de ejemplo.
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
            # Patrones de títulos profesionales
            patterns.extend([
                (r'\b(?:Dr|Dra|D|Dña|Dn|Dna|Sr|Sra|Srta|Prof|Enf|Aux|Lic|Ldo|Lda|R[1-5]|MIR|EIR|PIR)\b\.?', 
                 'Títulos profesionales abreviados'),
            ])
        
        elif category == "NUMEROS_IDENTIFICACION":
            # Patrones de números de identificación
            patterns.extend([
                (r'\b\d{8}[A-Z]\b', 'DNI español (8 dígitos + letra)'),
                (r'\b[XYZ]\d{7}[A-Z]\b', 'DNI con letra inicial'),
                (r'\b\d{9}\b', 'Teléfono de 9 dígitos'),
                (r'\+34\s*\d{1,3}\s*\d{3}\s*\d{2}\s*\d{2}', 'Teléfono con prefijo +34'),
                (r'\d{2,3}[-.\s]\d{3}[-.\s]\d{2}[-.\s]\d{2}', 'Teléfono con separadores'),
                (r'\b\d{5}\b', 'Código postal de 5 dígitos'),
                (r'\d{2}\s*\d{8}\s*\d{2}', 'Número de Seguridad Social'),
                (r'\d{2}[-]\d{8}[-]\d{2}', 'Número de Seguridad Social con guiones'),
            ])
        
        if patterns:
            patterns_dict[category] = patterns
    
    return patterns_dict

def find_entities_in_text(text: str, entities: List[str], fuzzy_threshold: float = 85.0, use_fuzzy: bool = True) -> List[str]:
    """
    Busca todas las entidades en el texto usando regex (búsqueda exacta) y fuzzy matching.
    """
    found_entities = []
    
    # Crear un conjunto de entidades únicas y ordenadas por longitud (más largas primero)
    unique_entities = [e for e in set(entities) if e and e.strip()]
    sorted_entities = sorted(unique_entities, key=len, reverse=True)
    
    # Diccionario para rastrear posiciones encontradas (para evitar solapamientos)
    found_positions = {}  # (start, end) -> entity
    
    # Buscar cada entidad
    for entity in sorted_entities:
        entity_lower = entity.lower().strip()
        entity_len = len(entity)
        
        # 1. Búsqueda exacta con regex
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
        if use_fuzzy and RAPIDFUZZ_AVAILABLE and entity_len >= 3:
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
    
    return found_entities

def find_patterns_in_text(text: str, patterns: List[Tuple[str, str]]) -> List[str]:
    """
    Busca patrones regex en el texto.
    TODAS las entidades deben tener espacio o puntuación antes y después.
    """
    found_entities = []
    found_positions = set()
    
    # Patrón para límites: espacio, inicio/fin de línea, o puntuación
    boundary_lookbehind = r'(?<![^\s\n\r\t.,;:!?()\[\]{}\'"])'
    boundary_lookahead = r'(?![^\s\n\r\t.,;:!?()\[\]{}\'"])'
    
    for pattern_str, description in patterns:
        try:
            # Agregar límites antes y después del patrón
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

def is_entity_included(entity1: str, entity2: str) -> bool:
    """
    Verifica si una entidad está incluida dentro de otra (case-insensitive).
    """
    if not entity1 or not entity2:
        return False
    
    entity1_lower = entity1.lower().strip()
    entity2_lower = entity2.lower().strip()
    
    return entity1_lower in entity2_lower or entity2_lower in entity1_lower

def calculate_strict_metrics(predicted: List[str], reference: List[str]) -> Dict:
    """
    Calcula métricas strict (coincidencia exacta, case-insensitive).
    """
    pred_normalized = [e.strip().lower() for e in predicted if e.strip()]
    ref_normalized = [e.strip().lower() for e in reference if e.strip()]
    
    pred_counter = Counter(pred_normalized)
    ref_counter = Counter(ref_normalized)
    
    true_positives = 0
    for entity, count in pred_counter.items():
        if entity in ref_counter:
            true_positives += min(count, ref_counter[entity])
    
    false_positives = sum(count for entity, count in pred_counter.items() 
                         if entity not in ref_counter)
    
    false_negatives = sum(count for entity, count in ref_counter.items() 
                         if entity not in pred_counter)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def calculate_relaxed_metrics(predicted: List[str], reference: List[str]) -> Dict:
    """
    Calcula métricas relaxed (entidad incluida dentro de otra cuenta como válida).
    """
    pred_list = [e.strip() for e in predicted if e.strip()]
    ref_list = [e.strip() for e in reference if e.strip()]
    
    pred_matched = [False] * len(pred_list)
    ref_matched = [False] * len(ref_list)
    
    true_positives = 0
    
    # Primero buscar coincidencias exactas
    for i, pred_entity in enumerate(pred_list):
        if pred_matched[i]:
            continue
        for j, ref_entity in enumerate(ref_list):
            if ref_matched[j]:
                continue
            if pred_entity.lower() == ref_entity.lower():
                true_positives += 1
                pred_matched[i] = True
                ref_matched[j] = True
                break
    
    # Luego buscar inclusiones
    for i, pred_entity in enumerate(pred_list):
        if pred_matched[i]:
            continue
        for j, ref_entity in enumerate(ref_list):
            if ref_matched[j]:
                continue
            if is_entity_included(pred_entity, ref_entity):
                true_positives += 1
                pred_matched[i] = True
                ref_matched[j] = True
                break
    
    false_positives = sum(1 for matched in pred_matched if not matched)
    false_negatives = sum(1 for matched in ref_matched if not matched)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def load_reference_entities(jsonl_file: str) -> Dict[str, List[str]]:
    """
    Carga las entidades de referencia desde un archivo JSONL.
    """
    entities_by_file = {}
    
    if not os.path.exists(jsonl_file):
        print(f"  ADVERTENCIA: No existe el archivo {jsonl_file}", flush=True)
        return entities_by_file
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    filename = entry.get('filename')
                    entities = entry.get('entities', [])
                    if filename:
                        entities_by_file[filename] = entities
                except json.JSONDecodeError as e:
                    print(f"  ADVERTENCIA: Error parseando línea: {e}", flush=True)
                    continue
    
    return entities_by_file

def main():
    """Función principal."""
    print("="*60, flush=True)
    print("EVALUACIÓN DE REGEX_EXTRACTION + PATTERN_EXTRACTION", flush=True)
    print("="*60, flush=True)
    
    # Rutas
    entities_file = "pipeline_anon/entities_reference.json"
    test_dir = "docs/test"
    reference_file = "docs/test/entities_test.jsonl"
    output_dir = "resultados"
    output_file = os.path.join(output_dir, "metricas_regex_pattern.json")
    
    # Crear directorio de resultados si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar entidades de referencia (para regex)
    print("\n1. Cargando entidades de referencia (regex)...", flush=True)
    start_time = time.time()
    try:
        entities_dict, entities_list = load_entities_reference(entities_file)
        print(f"  Total entidades únicas: {len(entities_list)}", flush=True)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar las entidades: {e}", flush=True)
        sys.exit(1)
    
    # Cargar formatos y generar patrones (para pattern extraction)
    print("\n2. Cargando formatos y generando patrones...", flush=True)
    try:
        formatos_dict = load_formatos_reference(entities_file)
        patterns_dict = generate_patterns_from_formatos(formatos_dict)
        total_patterns = sum(len(p) for p in patterns_dict.values())
        print(f"  Total patrones generados: {total_patterns}", flush=True)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los formatos: {e}", flush=True)
        sys.exit(1)
    
    load_time = time.time() - start_time
    print(f"  Tiempo de carga: {load_time:.2f} segundos", flush=True)
    
    # Cargar entidades de referencia (ground truth)
    print("\n3. Cargando entidades de referencia (ground truth)...", flush=True)
    reference_entities = load_reference_entities(reference_file)
    print(f"  Archivos de referencia: {len(reference_entities)}", flush=True)
    
    # Obtener lista de archivos .txt
    print(f"\n4. Buscando archivos .txt en {test_dir}...", flush=True)
    if not os.path.exists(test_dir):
        print(f"ERROR: No existe el directorio {test_dir}", flush=True)
        sys.exit(1)
    
    txt_files = list(Path(test_dir).glob("*.txt"))
    total_files_available = len(txt_files)
    print(f"  Archivos disponibles: {total_files_available}", flush=True)
    
    # Seleccionar 50 archivos aleatorios
    random.seed(42)  # Semilla para reproducibilidad
    files_to_process = random.sample(txt_files, min(50, total_files_available))
    total_files = len(files_to_process)
    
    print(f"  Archivos a procesar (aleatorios): {total_files}", flush=True)
    
    # Procesar archivos
    print("\n5. Procesando archivos...", flush=True)
    
    results = []
    all_strict_tp = 0
    all_strict_fp = 0
    all_strict_fn = 0
    all_relaxed_tp = 0
    all_relaxed_fp = 0
    all_relaxed_fn = 0
    
    processing_start = time.time()
    
    for idx, txt_file in enumerate(files_to_process, 1):
        filename = txt_file.name
        filename_normalized = filename.replace('.txt.txt', '.txt')
        
        # Leer texto
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"  ERROR leyendo {filename}: {e}", flush=True)
            continue
        
        # Extraer entidades con regex (step1)
        regex_entities = find_entities_in_text(text, entities_list, fuzzy_threshold=85.0, use_fuzzy=RAPIDFUZZ_AVAILABLE)
        
        # Extraer entidades con patrones (step2)
        pattern_entities = []
        for category, patterns in patterns_dict.items():
            category_entities = find_patterns_in_text(text, patterns)
            pattern_entities.extend(category_entities)
        
        # Combinar ambas listas (regex + patterns)
        predicted_entities = regex_entities + pattern_entities
        
        # Obtener entidades de referencia
        reference_entities_list = reference_entities.get(filename_normalized, [])
        if not reference_entities_list:
            reference_entities_list = reference_entities.get(filename, [])
        
        # Calcular métricas
        strict_metrics = calculate_strict_metrics(predicted_entities, reference_entities_list)
        relaxed_metrics = calculate_relaxed_metrics(predicted_entities, reference_entities_list)
        
        # Acumular métricas globales
        all_strict_tp += strict_metrics['true_positives']
        all_strict_fp += strict_metrics['false_positives']
        all_strict_fn += strict_metrics['false_negatives']
        all_relaxed_tp += relaxed_metrics['true_positives']
        all_relaxed_fp += relaxed_metrics['false_positives']
        all_relaxed_fn += relaxed_metrics['false_negatives']
        
        # Guardar resultados por archivo
        results.append({
            'filename': filename_normalized,
            'predicted_count': len(predicted_entities),
            'reference_count': len(reference_entities_list),
            'regex_count': len(regex_entities),
            'pattern_count': len(pattern_entities),
            'strict': strict_metrics,
            'relaxed': relaxed_metrics
        })
        
        # Mostrar progreso cada 10 archivos
        if idx % 10 == 0:
            print(f"  Procesados: {idx}/{total_files} archivos", flush=True)
    
    processing_time = time.time() - processing_start
    
    # Calcular métricas globales
    strict_precision = all_strict_tp / (all_strict_tp + all_strict_fp) if (all_strict_tp + all_strict_fp) > 0 else 0.0
    strict_recall = all_strict_tp / (all_strict_tp + all_strict_fn) if (all_strict_tp + all_strict_fn) > 0 else 0.0
    strict_f1 = 2 * (strict_precision * strict_recall) / (strict_precision + strict_recall) if (strict_precision + strict_recall) > 0 else 0.0
    
    relaxed_precision = all_relaxed_tp / (all_relaxed_tp + all_relaxed_fp) if (all_relaxed_tp + all_relaxed_fp) > 0 else 0.0
    relaxed_recall = all_relaxed_tp / (all_relaxed_tp + all_relaxed_fn) if (all_relaxed_tp + all_relaxed_fn) > 0 else 0.0
    relaxed_f1 = 2 * (relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall) if (relaxed_precision + relaxed_recall) > 0 else 0.0
    
    # Calcular métricas promedio por archivo
    if results:
        avg_strict_precision = sum(r['strict']['precision'] for r in results) / len(results)
        avg_strict_recall = sum(r['strict']['recall'] for r in results) / len(results)
        avg_strict_f1 = sum(r['strict']['f1'] for r in results) / len(results)
        
        avg_relaxed_precision = sum(r['relaxed']['precision'] for r in results) / len(results)
        avg_relaxed_recall = sum(r['relaxed']['recall'] for r in results) / len(results)
        avg_relaxed_f1 = sum(r['relaxed']['f1'] for r in results) / len(results)
    else:
        avg_strict_precision = avg_strict_recall = avg_strict_f1 = 0.0
        avg_relaxed_precision = avg_relaxed_recall = avg_relaxed_f1 = 0.0
    
    # Crear resumen
    summary = {
        'method': 'regex_extraction + pattern_extraction',
        'entities_reference_file': entities_file,
        'reference_file': reference_file,
        'test_directory': test_dir,
        'total_files_available': total_files_available,
        'total_files_processed': total_files,
        'files_selection': 'random_50',
        'load_time_seconds': load_time,
        'processing_time_seconds': processing_time,
        'total_time_seconds': load_time + processing_time,
        'fuzzy_matching_enabled': RAPIDFUZZ_AVAILABLE,
        'metrics': {
            'strict': {
                'precision': strict_precision,
                'recall': strict_recall,
                'f1': strict_f1,
                'true_positives': all_strict_tp,
                'false_positives': all_strict_fp,
                'false_negatives': all_strict_fn,
                'average_precision_per_file': avg_strict_precision,
                'average_recall_per_file': avg_strict_recall,
                'average_f1_per_file': avg_strict_f1
            },
            'relaxed': {
                'precision': relaxed_precision,
                'recall': relaxed_recall,
                'f1': relaxed_f1,
                'true_positives': all_relaxed_tp,
                'false_positives': all_relaxed_fp,
                'false_negatives': all_relaxed_fn,
                'average_precision_per_file': avg_relaxed_precision,
                'average_recall_per_file': avg_relaxed_recall,
                'average_f1_per_file': avg_relaxed_f1
            }
        },
        'per_file_results': results
    }
    
    # Guardar resultados
    print(f"\n6. Guardando resultados en {output_file}...", flush=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    print("\n" + "="*60, flush=True)
    print("RESUMEN DE MÉTRICAS", flush=True)
    print("="*60, flush=True)
    print(f"Archivos procesados: {total_files}", flush=True)
    print(f"Tiempo de carga: {load_time:.2f} segundos", flush=True)
    print(f"Tiempo de procesamiento: {processing_time:.2f} segundos", flush=True)
    print(f"Tiempo total: {load_time + processing_time:.2f} segundos", flush=True)
    print(f"Fuzzy matching: {'Habilitado' if RAPIDFUZZ_AVAILABLE else 'Deshabilitado'}", flush=True)
    print("\nMÉTRICAS STRICT (coincidencia exacta):", flush=True)
    print(f"  Precision: {strict_precision:.4f}", flush=True)
    print(f"  Recall: {strict_recall:.4f}", flush=True)
    print(f"  F1: {strict_f1:.4f}", flush=True)
    print(f"  TP: {all_strict_tp}, FP: {all_strict_fp}, FN: {all_strict_fn}", flush=True)
    print(f"  Promedio Precision por archivo: {avg_strict_precision:.4f}", flush=True)
    print(f"  Promedio Recall por archivo: {avg_strict_recall:.4f}", flush=True)
    print(f"  Promedio F1 por archivo: {avg_strict_f1:.4f}", flush=True)
    print("\nMÉTRICAS RELAXED (inclusión):", flush=True)
    print(f"  Precision: {relaxed_precision:.4f}", flush=True)
    print(f"  Recall: {relaxed_recall:.4f}", flush=True)
    print(f"  F1: {relaxed_f1:.4f}", flush=True)
    print(f"  TP: {all_relaxed_tp}, FP: {all_relaxed_fp}, FN: {all_relaxed_fn}", flush=True)
    print(f"  Promedio Precision por archivo: {avg_relaxed_precision:.4f}", flush=True)
    print(f"  Promedio Recall por archivo: {avg_relaxed_recall:.4f}", flush=True)
    print(f"  Promedio F1 por archivo: {avg_relaxed_f1:.4f}", flush=True)
    print("="*60, flush=True)
    print(f"\nResultados guardados en: {output_file}", flush=True)

if __name__ == "__main__":
    main()
