#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación del modelo bsc_ehr_anon_bin sobre documentos de test.

Este script evalúa el modelo bsc_ehr_anon_bin comparando sus predicciones
con las entidades de referencia en entities_test.jsonl y calcula métricas
strict y relaxed.
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Importar transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers no está disponible", flush=True)
    sys.exit(1)

def is_entity_included(entity1: str, entity2: str) -> bool:
    """
    Verifica si una entidad está incluida dentro de otra (case-insensitive).
    Usado para detectar solapamiento relaxed.
    
    Args:
        entity1: Primera entidad
        entity2: Segunda entidad
        
    Returns:
        True si una está incluida en la otra
    """
    if not entity1 or not entity2:
        return False
    
    entity1_lower = entity1.lower().strip()
    entity2_lower = entity2.lower().strip()
    
    return entity1_lower in entity2_lower or entity2_lower in entity1_lower

def load_model():
    """
    Carga el modelo BSC.
    
    Returns:
        Pipeline del modelo BSC
    """
    print("Cargando modelo BSC...", flush=True)
    
    # Configurar device: usar solo CPU (no GPU)
    device = -1  # -1 significa CPU
    print(f"  Device: cpu (forzado)", flush=True)
    
    # Modelo BSC
    bsc_model_path = "models/kbioxlm_anon_bin"
    
    try:
        bsc_tokenizer = AutoTokenizer.from_pretrained(bsc_model_path)
        # Configurar max_length en el tokenizer si no está definido
        if not hasattr(bsc_tokenizer, 'model_max_length') or bsc_tokenizer.model_max_length > 512:
            bsc_tokenizer.model_max_length = 512
        
        bsc_model = AutoModelForTokenClassification.from_pretrained(
            bsc_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
            trust_remote_code=False
        )
        # Asegurar que el modelo está en CPU
        bsc_model = bsc_model.to('cpu')
        
        pipeline_bsc = pipeline(
            "ner",
            model=bsc_model,
            tokenizer=bsc_tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        print("  Modelo BSC cargado exitosamente", flush=True)
        return pipeline_bsc
    except Exception as e:
        print(f"  ERROR cargando modelo BSC: {e}", flush=True)
        raise e

def extract_entities_with_model(text: str, pipeline_model) -> List[str]:
    """
    Extrae entidades de un texto usando el modelo BSC.
    Si el texto es muy largo, lo divide en chunks con solapamiento.
    Las entidades detectadas en zonas solapadas solo se cuentan una vez.
    
    Args:
        text: Texto a procesar
        pipeline_model: Pipeline del modelo
        
    Returns:
        Lista de entidades encontradas (solo texto, sin tipo de entidad)
    """
    entities = []
    
    try:
        max_chunk_length = 450
        
        if len(text) <= max_chunk_length:
            # Texto corto, procesar directamente
            model_entities = pipeline_model(text)
            
            if isinstance(model_entities, list):
                for entity in model_entities:
                    # Solo extraer el texto de la entidad, no el tipo
                    entity_text = entity.get('word', '').strip()
                    if entity_text:
                        entities.append(entity_text)
        else:
            # Texto largo, dividir en chunks con solapamiento
            chunk_size = max_chunk_length
            overlap_size = 100  # Aumentado para mejor cobertura
            
            # Diccionario para rastrear entidades ya encontradas en zonas solapadas
            # Clave: (texto_normalizado, posición_aproximada)
            seen_entities = {}
            
            i = 0
            chunk_num = 0
            max_chunks = 2000
            
            while i < len(text) and chunk_num < max_chunks:
                chunk_num += 1
                chunk_end = min(i + chunk_size, len(text))
                chunk = text[i:chunk_end]
                
                try:
                    chunk_entities = pipeline_model(chunk)
                    
                    if isinstance(chunk_entities, list):
                        for entity in chunk_entities:
                            # Solo extraer el texto de la entidad, no el tipo
                            entity_text = entity.get('word', '').strip()
                            
                            if not entity_text:
                                continue
                            
                            # Calcular posición absoluta en el texto original
                            entity_start_relative = entity.get('start', 0)
                            entity_start_absolute = i + entity_start_relative
                            
                            # Normalizar el texto de la entidad para comparación
                            entity_text_normalized = entity_text.lower().strip()
                            
                            # Crear clave para detectar duplicados en zona solapada
                            # Usar posición aproximada (redondeada a decenas) para agrupar entidades cercanas
                            position_key = (entity_start_absolute // 10) * 10
                            entity_key = (entity_text_normalized, position_key)
                            
                            # Verificar si ya vimos esta entidad en una zona solapada
                            if entity_key not in seen_entities:
                                # Nueva entidad, agregarla
                                entities.append(entity_text)
                                seen_entities[entity_key] = {
                                    'text': entity_text,
                                    'position': entity_start_absolute,
                                    'score': entity.get('score', 0.0)
                                }
                            else:
                                # Ya existe en zona solapada, mantener la de mayor confianza
                                existing_score = seen_entities[entity_key].get('score', 0.0)
                                current_score = entity.get('score', 0.0)
                                if current_score > existing_score:
                                    # Actualizar la entrada con mayor confianza
                                    seen_entities[entity_key]['score'] = current_score
                                    seen_entities[entity_key]['text'] = entity_text
                                    # Reemplazar en la lista de entidades
                                    # Buscar la última ocurrencia de esta entidad normalizada
                                    for idx in range(len(entities) - 1, -1, -1):
                                        if entities[idx].lower().strip() == entity_text_normalized:
                                            entities[idx] = entity_text
                                            break
                
                except Exception as e:
                    # Continuar con el siguiente chunk si hay error
                    pass
                
                # Avanzar con solapamiento
                i += chunk_size - overlap_size
                if i >= len(text):
                    break
        
        return entities
    
    except Exception as e:
        print(f"  ERROR extrayendo entidades: {e}", flush=True)
        return []

def load_reference_entities(jsonl_file: str) -> Dict[str, List[str]]:
    """
    Carga las entidades de referencia desde un archivo JSONL.
    
    Args:
        jsonl_file: Ruta al archivo JSONL
        
    Returns:
        Dict con filename como clave y lista de entidades como valor
    """
    entities_by_file = {}
    
    if not os.path.exists(jsonl_file):
        print(f"  ADVERTENCIA: No existe el archivo {jsonl_file}", flush=True)
        return entities_by_file
    
    print(f"Cargando entidades de referencia desde {jsonl_file}...", flush=True)
    
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
    
    print(f"  Archivos cargados: {len(entities_by_file)}", flush=True)
    return entities_by_file

def calculate_strict_metrics(predicted: List[str], reference: List[str]) -> Dict:
    """
    Calcula métricas strict (coincidencia exacta, case-insensitive).
    Misma implementación que validation.py.
    
    Args:
        predicted: Lista de entidades predichas
        reference: Lista de entidades de referencia
        
    Returns:
        Dict con precision, recall, f1, true_positives, false_positives, false_negatives
    """
    # Normalizar a minúsculas para comparación case-insensitive
    # Mantener el conteo pero usando la versión normalizada para comparar
    pred_normalized = [e.strip().lower() for e in predicted if e.strip()]
    ref_normalized = [e.strip().lower() for e in reference if e.strip()]
    
    # Convertir a contadores para tener en cuenta repeticiones
    pred_counter = Counter(pred_normalized)
    ref_counter = Counter(ref_normalized)
    
    # True Positives: entidades que aparecen en ambos con el mismo conteo mínimo
    true_positives = 0
    for entity, count in pred_counter.items():
        if entity in ref_counter:
            true_positives += min(count, ref_counter[entity])
    
    # False Positives: entidades predichas que no están en referencia
    false_positives = sum(count for entity, count in pred_counter.items() 
                         if entity not in ref_counter)
    
    # False Negatives: entidades de referencia que no fueron predichas
    false_negatives = sum(count for entity, count in ref_counter.items() 
                         if entity not in pred_counter)
    
    # Calcular métricas
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
    Misma implementación que validation.py.
    
    Args:
        predicted: Lista de entidades predichas
        reference: Lista de entidades de referencia
        
    Returns:
        Dict con precision, recall, f1, true_positives, false_positives, false_negatives
    """
    # Convertir a listas para mantener el orden y permitir repeticiones
    pred_list = [e.strip() for e in predicted if e.strip()]
    ref_list = [e.strip() for e in reference if e.strip()]
    
    # Crear listas de entidades ya emparejadas para evitar contar múltiples veces
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
    
    # Luego buscar inclusiones (predicted incluida en reference)
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
    
    # False Positives: entidades predichas que no fueron emparejadas
    false_positives = sum(1 for matched in pred_matched if not matched)
    
    # False Negatives: entidades de referencia que no fueron emparejadas
    false_negatives = sum(1 for matched in ref_matched if not matched)
    
    # Calcular métricas
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

def main():
    """Función principal."""
    print("="*60, flush=True)
    print("EVALUACIÓN DEL MODELO KBIOXLM_ANON_BIN", flush=True)
    print("="*60, flush=True)
    
    # Rutas
    test_dir = "docs/test"
    reference_file = "docs/test/entities_test.jsonl"
    output_dir = "resultados"
    output_file = os.path.join(output_dir, "metricas_kbioxlm_anon_bin.json")
    
    # Crear directorio de resultados si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar que existe el directorio de test
    if not os.path.exists(test_dir):
        print(f"ERROR: No existe el directorio {test_dir}", flush=True)
        sys.exit(1)
    
    # Cargar modelo
    start_time = time.time()
    pipeline_bsc = load_model()
    model_load_time = time.time() - start_time
    print(f"Tiempo de carga del modelo: {model_load_time:.2f} segundos", flush=True)
    
    # Cargar entidades de referencia
    reference_entities = load_reference_entities(reference_file)
    
    # Procesar archivos
    print("\nProcesando archivos de test...", flush=True)
    
    results = []
    all_strict_tp = 0
    all_strict_fp = 0
    all_strict_fn = 0
    all_relaxed_tp = 0
    all_relaxed_fp = 0
    all_relaxed_fn = 0
    
    txt_files = list(Path(test_dir).glob("*.txt"))
    total_files_available = len(txt_files)
    
    # Seleccionar 50 archivos aleatorios
    random.seed(42)  # Semilla para reproducibilidad
    files_to_process = random.sample(txt_files, min(50, total_files_available))
    total_files = len(files_to_process)
    
    print(f"Archivos disponibles: {total_files_available}", flush=True)
    print(f"Archivos a procesar (aleatorios): {total_files}", flush=True)
    
    processing_start = time.time()
    
    for idx, txt_file in enumerate(files_to_process, 1):
        filename = txt_file.name
        
        # Normalizar nombre de archivo para buscar en referencia
        # Remover .txt.txt si existe, dejar solo .txt
        filename_normalized = filename.replace('.txt.txt', '.txt')
        
        # Leer texto
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"  ERROR leyendo {filename}: {e}", flush=True)
            continue
        
        # Extraer entidades con el modelo
        predicted_entities = extract_entities_with_model(text, pipeline_bsc)
        
        # Obtener entidades de referencia usando nombre normalizado
        reference_entities_list = reference_entities.get(filename_normalized, [])
        
        # Si no se encuentra con nombre normalizado, intentar con el original
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
            'filename': filename,
            'predicted_count': len(predicted_entities),
            'reference_count': len(reference_entities_list),
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
        'model': 'kbioxlm_anon_bin',
        'test_directory': test_dir,
        'reference_file': reference_file,
        'total_files_available': total_files_available,
        'total_files_processed': total_files,
        'files_selection': 'random_50',
        'model_load_time_seconds': model_load_time,
        'processing_time_seconds': processing_time,
        'total_time_seconds': model_load_time + processing_time,
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
    print(f"\nGuardando resultados en {output_file}...", flush=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    print("\n" + "="*60, flush=True)
    print("RESUMEN DE MÉTRICAS", flush=True)
    print("="*60, flush=True)
    print(f"Archivos procesados: {total_files}", flush=True)
    print(f"Tiempo de carga del modelo: {model_load_time:.2f} segundos", flush=True)
    print(f"Tiempo de procesamiento: {processing_time:.2f} segundos", flush=True)
    print(f"Tiempo total: {model_load_time + processing_time:.2f} segundos", flush=True)
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

