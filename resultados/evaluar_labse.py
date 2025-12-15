#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para evaluar la detección de entidades usando embeddings LaBSE.
Usa similitud de coseno entre palabras del texto y vectores medios de subcategorías de entidades.
"""

import os
import sys
import json
import time
import random
import argparse
import re
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers no está instalado. Instálalo con: pip install sentence-transformers", flush=True)
    sys.exit(1)

def is_entity_included(entity1: str, entity2: str) -> bool:
    """
    Verifica si una entidad está incluida dentro de otra (relaxed matching).
    
    Args:
        entity1: Primera entidad
        entity2: Segunda entidad
        
    Returns:
        True si una está incluida en la otra
    """
    entity1_lower = entity1.strip().lower()
    entity2_lower = entity2.strip().lower()
    
    if not entity1_lower or not entity2_lower:
        return False
    
    return entity1_lower in entity2_lower or entity2_lower in entity1_lower

def load_entities_reference(json_file: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Carga entities_reference.json y agrupa por subcategorías.
    
    Args:
        json_file: Ruta al archivo JSON
        
    Returns:
        Dict con estructura: {categoria: {subcategoria: [entidades]}}
    """
    print(f"Cargando entidades de referencia desde {json_file}...", flush=True)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities = data.get('entities', {})
    
    # Agrupar por subcategorías (excluyendo "formatos" y "referencias")
    grouped = {}
    for category, subcategories in entities.items():
        grouped[category] = {}
        for subcat, entity_list in subcategories.items():
            if subcat not in ['formatos', 'referencias']:
                if isinstance(entity_list, list):
                    grouped[category][subcat] = entity_list
    
    total_subcats = sum(len(subs) for subs in grouped.values())
    total_entities = sum(len(ents) for subs in grouped.values() for ents in subs.values())
    print(f"  Categorías: {len(grouped)}", flush=True)
    print(f"  Subcategorías: {total_subcats}", flush=True)
    print(f"  Total entidades: {total_entities}", flush=True)
    
    return grouped

def compute_category_embeddings(model: SentenceTransformer, entities_grouped: Dict) -> Dict[str, np.ndarray]:
    """
    Calcula el embedding medio para cada subcategoría de entidades.
    
    Args:
        model: Modelo de embeddings
        entities_grouped: Entidades agrupadas por categoría y subcategoría
        
    Returns:
        Dict con {categoria/subcategoria: embedding_medio}
    """
    print("\nCalculando embeddings medios para cada subcategoría...", flush=True)
    
    category_embeddings = {}
    
    for category, subcategories in entities_grouped.items():
        for subcat, entity_list in subcategories.items():
            if not entity_list:
                continue
            
            key = f"{category}/{subcat}"
            print(f"  Procesando {key} ({len(entity_list)} entidades)...", flush=True)
            
            # Calcular embeddings para todas las entidades de esta subcategoría
            embeddings = model.encode(entity_list, show_progress_bar=False, batch_size=32)
            
            # Calcular el embedding medio
            mean_embedding = np.mean(embeddings, axis=0)
            category_embeddings[key] = mean_embedding
    
    print(f"  Total subcategorías procesadas: {len(category_embeddings)}", flush=True)
    return category_embeddings

def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokeniza el texto en palabras, preservando posiciones.
    
    Args:
        text: Texto a tokenizar
        
    Returns:
        Lista de tuplas (palabra, inicio, fin)
    """
    words = []
    # Usar regex para encontrar palabras (letras, números, algunos caracteres especiales)
    pattern = r'\b\w+\b'
    for match in re.finditer(pattern, text):
        word = match.group()
        start = match.start()
        end = match.end()
        words.append((word, start, end))
    
    return words

def extract_entities_with_embeddings(
    text: str,
    model: SentenceTransformer,
    category_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.7
) -> List[str]:
    """
    Extrae entidades del texto usando similitud de coseno con embeddings.
    
    Args:
        text: Texto a procesar
        model: Modelo de embeddings
        category_embeddings: Embeddings medios por subcategoría
        threshold: Umbral de similitud de coseno
        
    Returns:
        Lista de entidades detectadas
    """
    entities = []
    
    # Tokenizar texto
    words = tokenize_text(text)
    
    if not words:
        return entities
    
    # Calcular embeddings para todas las palabras
    word_texts = [word for word, _, _ in words]
    word_embeddings = model.encode(word_texts, show_progress_bar=False, batch_size=64)
    
    # Para cada subcategoría, calcular similitud con cada palabra
    for category_key, category_embedding in category_embeddings.items():
        # Calcular similitud de coseno entre palabras y el embedding medio
        similarities = cosine_similarity(word_embeddings, category_embedding.reshape(1, -1)).flatten()
        
        # Filtrar palabras con similitud > threshold
        for i, (word, start, end) in enumerate(words):
            if similarities[i] > threshold:
                # Verificar que la palabra tenga espacio o puntuación antes y después
                if start > 0 and text[start-1] not in ' \n\t.,;:!?()[]{}':
                    continue
                if end < len(text) and text[end] not in ' \n\t.,;:!?()[]{}':
                    continue
                
                entities.append(word)
    
    return entities

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
                        # Normalizar nombre de archivo (.txt.txt -> .txt)
                        if filename.endswith('.txt.txt'):
                            filename = filename[:-4]
                        entities_by_file[filename] = entities
                except json.JSONDecodeError as e:
                    print(f"  ADVERTENCIA: Error parseando línea: {e}", flush=True)
                    continue
    
    print(f"  Archivos cargados: {len(entities_by_file)}", flush=True)
    return entities_by_file

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
    
    # Coincidencias exactas
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
    
    # Inclusiones
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

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Evaluar detección de entidades con LaBSE')
    parser.add_argument('--test-dir', type=str, default='docs/train',
                        help='Directorio con archivos de texto (default: docs/train)')
    parser.add_argument('--reference-file', type=str, default='docs/train/entities_train.jsonl',
                        help='Archivo JSONL con entidades de referencia (default: docs/train/entities_train.jsonl)')
    parser.add_argument('--entities-ref', type=str, default='pipeline_anon/entities_reference.json',
                        help='Archivo JSON con entidades de referencia (default: pipeline_anon/entities_reference.json)')
    parser.add_argument('--model-path', type=str, default='models/labse',
                        help='Ruta al modelo LaBSE (default: models/labse)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Umbral de similitud de coseno (default: 0.7)')
    parser.add_argument('--num-files', type=int, default=50,
                        help='Número de archivos aleatorios a procesar (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para aleatoriedad (default: 42)')
    parser.add_argument('--output-dir', type=str, default='resultados',
                        help='Directorio de salida (default: resultados)')
    parser.add_argument('--output-file', type=str, default='metricas_labse.json',
                        help='Nombre del archivo de salida (default: metricas_labse.json)')
    
    args = parser.parse_args()
    
    print("="*60, flush=True)
    print("EVALUACIÓN CON EMBEDDINGS LABSE", flush=True)
    print("="*60, flush=True)
    
    # Crear directorio de resultados
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # Verificar directorio de test
    if not os.path.exists(args.test_dir):
        print(f"ERROR: No existe el directorio {args.test_dir}", flush=True)
        sys.exit(1)
    
    # Cargar modelo
    print(f"\nCargando modelo LaBSE desde {args.model_path}...", flush=True)
    start_time = time.time()
    try:
        model = SentenceTransformer(args.model_path)
        model_load_time = time.time() - start_time
        print(f"  Modelo cargado en {model_load_time:.2f} segundos", flush=True)
    except Exception as e:
        print(f"  ERROR cargando modelo: {e}", flush=True)
        sys.exit(1)
    
    # Cargar entidades de referencia
    entities_grouped = load_entities_reference(args.entities_ref)
    
    # Calcular embeddings medios por subcategoría
    category_embeddings = compute_category_embeddings(model, entities_grouped)
    
    # Cargar entidades de referencia (ground truth)
    reference_entities = load_reference_entities(args.reference_file)
    
    # Obtener archivos de texto
    txt_files = [f for f in os.listdir(args.test_dir) if f.endswith('.txt')]
    
    # Normalizar nombres de archivo
    txt_files_normalized = []
    for f in txt_files:
        if f.endswith('.txt.txt'):
            normalized = f[:-4]
        else:
            normalized = f
        txt_files_normalized.append((f, normalized))
    
    # Seleccionar archivos aleatorios
    random.seed(args.seed)
    if len(txt_files_normalized) > args.num_files:
        selected = random.sample(txt_files_normalized, args.num_files)
    else:
        selected = txt_files_normalized
    
    print(f"\nProcesando {len(selected)} archivos...", flush=True)
    
    # Procesar archivos
    results = []
    total_strict_tp = 0
    total_strict_fp = 0
    total_strict_fn = 0
    total_relaxed_tp = 0
    total_relaxed_fp = 0
    total_relaxed_fn = 0
    
    start_process_time = time.time()
    
    for i, (filename, normalized_filename) in enumerate(selected, 1):
        filepath = os.path.join(args.test_dir, filename)
        
        # Leer texto
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"  ERROR leyendo {filename}: {e}", flush=True)
            continue
        
        # Extraer entidades
        predicted_entities = extract_entities_with_embeddings(
            text, model, category_embeddings, args.threshold
        )
        
        # Obtener entidades de referencia
        reference = reference_entities.get(normalized_filename, [])
        
        # Calcular métricas
        strict_metrics = calculate_strict_metrics(predicted_entities, reference)
        relaxed_metrics = calculate_relaxed_metrics(predicted_entities, reference)
        
        # Acumular totales
        total_strict_tp += strict_metrics['true_positives']
        total_strict_fp += strict_metrics['false_positives']
        total_strict_fn += strict_metrics['false_negatives']
        total_relaxed_tp += relaxed_metrics['true_positives']
        total_relaxed_fp += relaxed_metrics['false_positives']
        total_relaxed_fn += relaxed_metrics['false_negatives']
        
        # Guardar resultados por archivo
        results.append({
            'filename': normalized_filename,
            'predicted_count': len(predicted_entities),
            'reference_count': len(reference),
            'strict': strict_metrics,
            'relaxed': relaxed_metrics
        })
        
        if i % 10 == 0:
            print(f"  Procesados {i}/{len(selected)} archivos...", flush=True)
    
    process_time = time.time() - start_process_time
    
    # Calcular métricas globales
    strict_precision = total_strict_tp / (total_strict_tp + total_strict_fp) if (total_strict_tp + total_strict_fp) > 0 else 0.0
    strict_recall = total_strict_tp / (total_strict_tp + total_strict_fn) if (total_strict_tp + total_strict_fn) > 0 else 0.0
    strict_f1 = 2 * (strict_precision * strict_recall) / (strict_precision + strict_recall) if (strict_precision + strict_recall) > 0 else 0.0
    
    relaxed_precision = total_relaxed_tp / (total_relaxed_tp + total_relaxed_fp) if (total_relaxed_tp + total_relaxed_fp) > 0 else 0.0
    relaxed_recall = total_relaxed_tp / (total_relaxed_tp + total_relaxed_fn) if (total_relaxed_tp + total_relaxed_fn) > 0 else 0.0
    relaxed_f1 = 2 * (relaxed_precision * relaxed_recall) / (relaxed_precision + relaxed_recall) if (relaxed_precision + relaxed_recall) > 0 else 0.0
    
    # Calcular métricas promedio por archivo
    avg_strict_precision = np.mean([r['strict']['precision'] for r in results])
    avg_strict_recall = np.mean([r['strict']['recall'] for r in results])
    avg_strict_f1 = np.mean([r['strict']['f1'] for r in results])
    
    avg_relaxed_precision = np.mean([r['relaxed']['precision'] for r in results])
    avg_relaxed_recall = np.mean([r['relaxed']['recall'] for r in results])
    avg_relaxed_f1 = np.mean([r['relaxed']['f1'] for r in results])
    
    # Crear resumen
    summary = {
        'model': 'LaBSE',
        'test_dir': args.test_dir,
        'reference_file': args.reference_file,
        'entities_ref': args.entities_ref,
        'threshold': args.threshold,
        'num_files_processed': len(results),
        'model_load_time': model_load_time,
        'processing_time': process_time,
        'total_time': model_load_time + process_time,
        'strict_metrics': {
            'precision': strict_precision,
            'recall': strict_recall,
            'f1': strict_f1,
            'true_positives': total_strict_tp,
            'false_positives': total_strict_fp,
            'false_negatives': total_strict_fn
        },
        'relaxed_metrics': {
            'precision': relaxed_precision,
            'recall': relaxed_recall,
            'f1': relaxed_f1,
            'true_positives': total_relaxed_tp,
            'false_positives': total_relaxed_fp,
            'false_negatives': total_relaxed_fn
        },
        'avg_per_file': {
            'strict': {
                'precision': avg_strict_precision,
                'recall': avg_strict_recall,
                'f1': avg_strict_f1
            },
            'relaxed': {
                'precision': avg_relaxed_precision,
                'recall': avg_relaxed_recall,
                'f1': avg_relaxed_f1
            }
        },
        'per_file_results': results
    }
    
    # Guardar resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    print("\n" + "="*60, flush=True)
    print("RESUMEN DE RESULTADOS", flush=True)
    print("="*60, flush=True)
    print(f"\nArchivos procesados: {len(results)}", flush=True)
    print(f"Tiempo total: {model_load_time + process_time:.2f} segundos", flush=True)
    print(f"\nMÉTRICAS STRICT:", flush=True)
    print(f"  Precision: {strict_precision:.4f}", flush=True)
    print(f"  Recall: {strict_recall:.4f}", flush=True)
    print(f"  F1: {strict_f1:.4f}", flush=True)
    print(f"  TP: {total_strict_tp}, FP: {total_strict_fp}, FN: {total_strict_fn}", flush=True)
    print(f"\nMÉTRICAS RELAXED:", flush=True)
    print(f"  Precision: {relaxed_precision:.4f}", flush=True)
    print(f"  Recall: {relaxed_recall:.4f}", flush=True)
    print(f"  F1: {relaxed_f1:.4f}", flush=True)
    print(f"  TP: {total_relaxed_tp}, FP: {total_relaxed_fp}, FN: {total_relaxed_fn}", flush=True)
    print(f"\nResultados guardados en: {output_file}", flush=True)

if __name__ == '__main__':
    main()

