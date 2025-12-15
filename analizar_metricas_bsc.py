#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis detallado de las métricas del modelo BSC para entender el bajo recall.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def analyze_file_entities(predicted: list, reference: list, filename: str):
    """Analiza las entidades de un archivo para entender discrepancias."""
    pred_normalized = [e.strip().lower() for e in predicted if e.strip()]
    ref_normalized = [e.strip().lower() for e in reference if e.strip()]
    
    pred_counter = Counter(pred_normalized)
    ref_counter = Counter(ref_normalized)
    
    # Entidades que el modelo detectó pero no están en referencia
    false_positives = set(pred_counter.keys()) - set(ref_counter.keys())
    
    # Entidades que están en referencia pero el modelo no detectó
    false_negatives = set(ref_counter.keys()) - set(pred_counter.keys())
    
    # Entidades que coinciden
    true_positives = set(pred_counter.keys()) & set(ref_counter.keys())
    
    return {
        'filename': filename,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_positives': true_positives,
        'pred_unique': len(pred_counter),
        'ref_unique': len(ref_counter)
    }

def main():
    """Función principal."""
    print("="*60, flush=True)
    print("ANÁLISIS DE MÉTRICAS BSC", flush=True)
    print("="*60, flush=True)
    
    # Cargar métricas
    with open('resultados/metricas_bsc_model.json', 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
    
    # Cargar entidades de referencia
    reference_entities = {}
    with open('docs/test/entities_test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            reference_entities[entry['filename']] = entry['entities']
    
    print(f"\nMétricas globales:", flush=True)
    print(f"  Strict - Precision: {metrics_data['metrics']['strict']['precision']:.4f}", flush=True)
    print(f"  Strict - Recall: {metrics_data['metrics']['strict']['recall']:.4f}", flush=True)
    print(f"  Strict - F1: {metrics_data['metrics']['strict']['f1']:.4f}", flush=True)
    print(f"  Strict - TP: {metrics_data['metrics']['strict']['true_positives']}", flush=True)
    print(f"  Strict - FP: {metrics_data['metrics']['strict']['false_positives']}", flush=True)
    print(f"  Strict - FN: {metrics_data['metrics']['strict']['false_negatives']}", flush=True)
    
    # Analizar algunos archivos con más falsos negativos
    print("\n" + "="*60, flush=True)
    print("Archivos con más falsos negativos (strict):", flush=True)
    print("="*60, flush=True)
    
    files_by_fn = sorted(
        metrics_data['per_file_results'],
        key=lambda x: x['strict']['false_negatives'],
        reverse=True
    )[:10]
    
    all_false_negatives = Counter()
    all_false_positives = Counter()
    
    for file_data in files_by_fn:
        filename = file_data['filename']
        fn_count = file_data['strict']['false_negatives']
        fp_count = file_data['strict']['false_positives']
        
        print(f"\nArchivo: {filename}", flush=True)
        print(f"  Predichas: {file_data['predicted_count']}, Referencia: {file_data['reference_count']}", flush=True)
        print(f"  FN: {fn_count}, FP: {fp_count}", flush=True)
        print(f"  Recall: {file_data['strict']['recall']:.4f}", flush=True)
        
        # Analizar entidades específicas
        if filename in reference_entities:
            ref_entities = reference_entities[filename]
            # Necesitamos las entidades predichas del modelo
            # Por ahora solo mostramos estadísticas
            
    # Analizar patrones globales de falsos negativos
    print("\n" + "="*60, flush=True)
    print("Análisis de patrones globales", flush=True)
    print("="*60, flush=True)
    
    # Contar tipos de entidades en referencia que no se detectan
    ref_all = []
    pred_all = []
    
    for file_data in metrics_data['per_file_results']:
        filename = file_data['filename']
        if filename in reference_entities:
            ref_all.extend([e.lower().strip() for e in reference_entities[filename]])
            # No tenemos las entidades predichas aquí, solo contamos
    
    ref_counter = Counter(ref_all)
    
    print(f"\nTotal entidades únicas en referencia: {len(ref_counter)}", flush=True)
    print(f"Top 20 entidades más frecuentes en referencia:", flush=True)
    for entity, count in ref_counter.most_common(20):
        print(f"  {entity}: {count}", flush=True)
    
    # Verificar si el problema es con nombres de archivos
    print("\n" + "="*60, flush=True)
    print("Verificación de nombres de archivos", flush=True)
    print("="*60, flush=True)
    
    # Comparar nombres en métricas vs referencia
    metrics_filenames = set(f['filename'] for f in metrics_data['per_file_results'])
    ref_filenames = set(reference_entities.keys())
    
    print(f"Archivos en métricas: {len(metrics_filenames)}", flush=True)
    print(f"Archivos en referencia: {len(ref_filenames)}", flush=True)
    
    missing_in_metrics = ref_filenames - metrics_filenames
    missing_in_ref = metrics_filenames - ref_filenames
    
    if missing_in_metrics:
        print(f"\nArchivos en referencia pero no en métricas ({len(missing_in_metrics)}):", flush=True)
        for fname in list(missing_in_metrics)[:5]:
            print(f"  {fname}", flush=True)
    
    if missing_in_ref:
        print(f"\nArchivos en métricas pero no en referencia ({len(missing_in_ref)}):", flush=True)
        for fname in list(missing_in_ref)[:5]:
            print(f"  {fname}", flush=True)
    
    # Verificar diferencias en nombres (extensión .txt.txt vs .txt)
    print("\nVerificando diferencias en extensiones de archivos...", flush=True)
    metrics_no_ext = set(f.replace('.txt.txt', '').replace('.txt', '') for f in metrics_filenames)
    ref_no_ext = set(f.replace('.txt.txt', '').replace('.txt', '') for f in ref_filenames)
    
    if metrics_no_ext == ref_no_ext:
        print("  ✓ Los nombres de archivos coinciden (ignorando extensiones)", flush=True)
    else:
        print("  ✗ Hay diferencias en nombres de archivos", flush=True)
        diff = metrics_no_ext - ref_no_ext
        if diff:
            print(f"  En métricas pero no en ref: {len(diff)}", flush=True)
        diff = ref_no_ext - metrics_no_ext
        if diff:
            print(f"  En ref pero no en métricas: {len(diff)}", flush=True)

if __name__ == "__main__":
    main()


