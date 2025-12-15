#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Anonimización - Paso 4: Extracción de entidades con modelos BSC y KBio

Este script usa los modelos bsc_ehr_anon_bin y kbioxlm_anon_bin para detectar entidades
en los archivos de texto. Las nuevas entidades se agregan a entities_anon.jsonl según
criterios de confianza:
- Si ambos modelos detectan la misma entidad (solapamiento relaxed): ambos >= 0.8
- Si solo un modelo detecta: confianza >= 0.98
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import multiprocessing
import threading

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Importar transformers solo cuando sea necesario (en el worker)
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ADVERTENCIA: transformers no está disponible", flush=True)

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

def load_models():
    """
    Carga los modelos BSC y KBio.
    
    Returns:
        Tuple: (pipeline_bsc, pipeline_kbio)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers no está disponible")
    
    print("Cargando modelos BSC y KBio...", flush=True)
    
    # Configurar device
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    
    # Modelo BSC
    print("  - Cargando bsc_ehr_anon_bin...", flush=True)
    bsc_model_path = "models/bsc_ehr_anon_bin"
    
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
        
        pipeline_bsc = pipeline(
            "ner",
            model=bsc_model,
            tokenizer=bsc_tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        print("    BSC cargado exitosamente", flush=True)
    except Exception as e:
        print(f"    ERROR cargando BSC: {e}", flush=True)
        raise e
    
    # Modelo KBio
    print("  - Cargando kbioxlm_anon_bin...", flush=True)
    kbio_model_path = "models/kbioxlm_anon_bin"
    
    try:
        kbio_tokenizer = AutoTokenizer.from_pretrained(kbio_model_path)
        # Configurar max_length en el tokenizer si no está definido
        if not hasattr(kbio_tokenizer, 'model_max_length') or kbio_tokenizer.model_max_length > 512:
            kbio_tokenizer.model_max_length = 512
        
        kbio_model = AutoModelForTokenClassification.from_pretrained(
            kbio_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
            trust_remote_code=False
        )
        
        pipeline_kbio = pipeline(
            "ner",
            model=kbio_model,
            tokenizer=kbio_tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        print("    KBio cargado exitosamente", flush=True)
    except Exception as e:
        print(f"    ERROR cargando KBio: {e}", flush=True)
        raise e
    
    print("Modelos cargados exitosamente!", flush=True)
    return pipeline_bsc, pipeline_kbio

def extract_entities_with_model(text: str, pipeline_model, model_name: str) -> List[Dict]:
    """
    Extrae entidades de un texto usando un modelo específico.
    Si el texto es muy largo, lo divide en chunks con solapamiento.
    Las entidades detectadas en zonas solapadas solo se cuentan una vez.
    
    Args:
        text: Texto a procesar
        pipeline_model: Pipeline del modelo
        model_name: Nombre del modelo para logging
        
    Returns:
        Lista de entidades encontradas con formato: {
            'word': texto de la entidad,
            'start': posición inicio,
            'end': posición fin,
            'score': confianza,
            'model': nombre del modelo
        }
    """
    entities = []
    
    try:
        # Longitud máxima aproximada de tokens (los modelos suelen tener límite de 512)
        # Aumentar el tamaño del chunk para reducir número de llamadas al modelo
        max_chunk_length = 450  # Aumentado para procesar más texto por llamada (menos llamadas = más rápido)
        
        if len(text) <= max_chunk_length:
            # Texto corto, procesar directamente sin timeout (más rápido)
            try:
                model_entities = pipeline_model(text)
                
                if isinstance(model_entities, list):
                    for entity in model_entities:
                        entity_text = entity.get('word', '').strip()
                        entity_score = entity.get('score', 0.0)
                        
                        if entity_text:
                            entities.append({
                                'word': entity_text,
                                'start': entity.get('start', 0),
                                'end': entity.get('end', 0),
                                'score': float(entity_score),
                                'model': model_name
                            })
            except (TimeoutError, Exception) as e:
                # Si hay error o timeout, continuar sin entidades de este modelo
                pass
        else:
            # Texto largo, dividir en chunks con solapamiento
            # Optimización: dividir por palabras para evitar cortar palabras a la mitad
            chunk_size = max_chunk_length
            overlap_size = 100  # Solapamiento razonable para no perder entidades
            
            # Dividir por caracteres (más simple y rápido que por palabras)
            # Diccionario para rastrear entidades ya encontradas (más eficiente)
            seen_entities = {}
            
            chunk_num = 0
            max_chunks = 2000  # Límite de chunks para evitar loops infinitos
            
            i = 0
            while i < len(text) and chunk_num < max_chunks:
                chunk_num += 1
                # Calcular fin del chunk
                chunk_end = min(i + chunk_size, len(text))
                chunk = text[i:chunk_end]
                
                try:
                    # Procesar sin timeout (más rápido, confiamos en que los modelos funcionan)
                    chunk_entities = pipeline_model(chunk)
                    
                    if isinstance(chunk_entities, list):
                        for entity in chunk_entities:
                            entity_text = entity.get('word', '').strip()
                            entity_score = entity.get('score', 0.0)
                            
                            if not entity_text:
                                continue
                            
                            # Calcular posición absoluta en el texto original
                            entity_start_relative = entity.get('start', 0)
                            entity_end_relative = entity.get('end', 0)
                            entity_start_absolute = i + entity_start_relative
                            entity_end_absolute = i + entity_end_relative
                            
                            # Normalizar el texto de la entidad para comparación
                            entity_text_normalized = entity_text.lower().strip()
                            
                            # Crear clave para detectar duplicados en zona solapada
                            position_key = (entity_start_absolute // 10) * 10
                            entity_key = (entity_text_normalized, position_key)
                            
                            # Verificar si ya vimos esta entidad en una zona solapada
                            if entity_key in seen_entities:
                                # Ya existe, mantener la de mayor confianza
                                existing_score = seen_entities[entity_key]['score']
                                if entity_score > existing_score:
                                    # Reemplazar con la de mayor confianza
                                    existing_idx = seen_entities[entity_key]['index']
                                    entities[existing_idx] = {
                                        'word': entity_text,
                                        'start': entity_start_absolute,
                                        'end': entity_end_absolute,
                                        'score': float(entity_score),
                                        'model': model_name
                                    }
                                    seen_entities[entity_key]['score'] = entity_score
                            else:
                                # Nueva entidad, agregarla
                                entity_dict = {
                                    'word': entity_text,
                                    'start': entity_start_absolute,
                                    'end': entity_end_absolute,
                                    'score': float(entity_score),
                                    'model': model_name
                                }
                                entities.append(entity_dict)
                                seen_entities[entity_key] = {
                                    'score': entity_score,
                                    'index': len(entities) - 1
                                }
                                
                except (TimeoutError, Exception) as e:
                    # Continuar con el siguiente chunk si hay error o timeout
                    # No imprimir error para no saturar la salida
                    pass
                
                # Avanzar al siguiente chunk con solapamiento
                if chunk_end >= len(text):
                    break
                i += chunk_size - overlap_size
    
    except Exception as e:
        # Si hay error general, devolver lista vacía
        pass
    
    return entities

def find_overlapping_entities(bsc_entities: List[Dict], kbio_entities: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Encuentra entidades que se solapan entre ambos modelos (relaxed matching).
    
    Args:
        bsc_entities: Lista de entidades detectadas por BSC
        kbio_entities: Lista de entidades detectadas por KBio
        
    Returns:
        Lista de tuplas (entidad_bsc, entidad_kbio) que se solapan
    """
    overlapping = []
    kbio_matched = [False] * len(kbio_entities)
    
    for bsc_entity in bsc_entities:
        bsc_text = bsc_entity.get('word', '').strip()
        if not bsc_text:
            continue
        
        for i, kbio_entity in enumerate(kbio_entities):
            if kbio_matched[i]:
                continue
            
            kbio_text = kbio_entity.get('word', '').strip()
            if not kbio_text:
                continue
            
            # Verificar solapamiento relaxed (inclusión)
            if is_entity_included(bsc_text, kbio_text):
                overlapping.append((bsc_entity, kbio_entity))
                kbio_matched[i] = True
                break
    
    return overlapping

def is_x_entity(text: str) -> bool:
    """
    Verifica si una entidad contiene o es solo caracteres X (XXX, XXXX, etc.).
    
    Args:
        text: Texto a verificar
        
    Returns:
        True si contiene o es solo X's
    """
    if not text:
        return False
    
    text_upper = text.upper().strip()
    # Verificar si contiene XXX o es solo X's
    if 'XXX' in text_upper:
        return True
    
    # Verificar si es solo caracteres X (y espacios)
    cleaned = text_upper.replace(' ', '').replace('\t', '').replace('\n', '')
    if len(cleaned) > 0 and all(c == 'X' for c in cleaned):
        return True
    
    return False

def filter_entities_by_confidence(bsc_entities: List[Dict], kbio_entities: List[Dict],
                                 both_threshold: float = 0.8, single_threshold: float = 0.98,
                                 debug_filename: str = None) -> List[str]:
    """
    Filtra entidades según los criterios de confianza:
    - Si ambos modelos detectan la misma entidad: ambos >= both_threshold
    - Si solo un modelo detecta: confianza >= single_threshold
    - Descarta entidades que contengan o sean XXX
    
    Args:
        bsc_entities: Lista de entidades detectadas por BSC
        kbio_entities: Lista de entidades detectadas por KBio
        both_threshold: Umbral cuando ambos modelos detectan (default: 0.8)
        single_threshold: Umbral cuando solo un modelo detecta (default: 0.98)
        debug_filename: Nombre del archivo para debug (opcional)
        
    Returns:
        Lista de entidades (texto) que cumplen los criterios
    """
    accepted_entities = []
    debug_mode = debug_filename and debug_filename.startswith("0000000010000000011219557")
    
    if debug_mode:
        print(f"      FILTRO: BSC={len(bsc_entities)} entidades, KBio={len(kbio_entities)} entidades", flush=True)
    
    # Encontrar entidades que se solapan
    overlapping = find_overlapping_entities(bsc_entities, kbio_entities)
    
    if debug_mode:
        print(f"      FILTRO: {len(overlapping)} entidades solapadas encontradas", flush=True)
    
    # Procesar entidades solapadas
    bsc_matched = set()
    kbio_matched = set()
    rejected_both_low_conf = 0
    rejected_x_entity = 0
    
    for bsc_entity, kbio_entity in overlapping:
        bsc_score = bsc_entity.get('score', 0.0)
        kbio_score = kbio_entity.get('score', 0.0)
        
        # Si ambos tienen confianza >= both_threshold, aceptar
        if bsc_score >= both_threshold and kbio_score >= both_threshold:
            # Usar el texto más largo o el de mayor confianza
            bsc_text = bsc_entity.get('word', '').strip()
            kbio_text = kbio_entity.get('word', '').strip()
            
            selected_text = kbio_text if len(kbio_text) > len(bsc_text) else bsc_text
            
            # Filtrar entidades que contengan o sean XXX
            if not is_x_entity(selected_text):
                accepted_entities.append(selected_text)
                if debug_mode and len(accepted_entities) <= 3:
                    print(f"        Aceptada (ambos): '{selected_text[:40]}' (BSC={bsc_score:.3f}, KBio={kbio_score:.3f})", flush=True)
            else:
                rejected_x_entity += 1
                if debug_mode:
                    print(f"        Rechazada (XXX): '{selected_text[:40]}'", flush=True)
        else:
            rejected_both_low_conf += 1
            if debug_mode:
                bsc_text = bsc_entity.get('word', '').strip()[:40]
                print(f"        Rechazada (confianza baja): '{bsc_text}' (BSC={bsc_score:.3f}, KBio={kbio_score:.3f})", flush=True)
        
        bsc_matched.add(id(bsc_entity))
        kbio_matched.add(id(kbio_entity))
    
    # Procesar entidades no solapadas de BSC
    rejected_single_low_conf_bsc = 0
    for bsc_entity in bsc_entities:
        if id(bsc_entity) in bsc_matched:
            continue
        
        bsc_score = bsc_entity.get('score', 0.0)
        if bsc_score >= single_threshold:
            bsc_text = bsc_entity.get('word', '').strip()
            if bsc_text and not is_x_entity(bsc_text):
                accepted_entities.append(bsc_text)
                if debug_mode and len(accepted_entities) <= 5:
                    print(f"        Aceptada (solo BSC): '{bsc_text[:40]}' (conf={bsc_score:.3f})", flush=True)
            else:
                rejected_x_entity += 1
        else:
            rejected_single_low_conf_bsc += 1
    
    # Procesar entidades no solapadas de KBio
    rejected_single_low_conf_kbio = 0
    for kbio_entity in kbio_entities:
        if id(kbio_entity) in kbio_matched:
            continue
        
        kbio_score = kbio_entity.get('score', 0.0)
        if kbio_score >= single_threshold:
            kbio_text = kbio_entity.get('word', '').strip()
            if kbio_text and not is_x_entity(kbio_text):
                accepted_entities.append(kbio_text)
                if debug_mode and len(accepted_entities) <= 5:
                    print(f"        Aceptada (solo KBio): '{kbio_text[:40]}' (conf={kbio_score:.3f})", flush=True)
            else:
                rejected_x_entity += 1
        else:
            rejected_single_low_conf_kbio += 1
    
    if debug_mode:
        print(f"      FILTRO: Aceptadas={len(accepted_entities)}, Rechazadas confianza baja (ambos)={rejected_both_low_conf}, "
              f"Rechazadas confianza baja (solo BSC)={rejected_single_low_conf_bsc}, "
              f"Rechazadas confianza baja (solo KBio)={rejected_single_low_conf_kbio}, "
              f"Rechazadas XXX={rejected_x_entity}", flush=True)
    
    # Eliminar duplicados manteniendo el orden
    return list(dict.fromkeys(accepted_entities))

def process_text_file_with_models(file_path: str, pipeline_bsc, pipeline_kbio,
                                  both_threshold: float = 0.8, single_threshold: float = 0.98) -> Dict:
    """
    Procesa un archivo de texto con ambos modelos y extrae entidades.
    
    Args:
        file_path: Ruta al archivo de texto
        pipeline_bsc: Pipeline del modelo BSC
        pipeline_kbio: Pipeline del modelo KBio
        both_threshold: Umbral cuando ambos modelos detectan
        single_threshold: Umbral cuando solo un modelo detecta
        
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
    
    # Extraer entidades con ambos modelos
    bsc_entities = extract_entities_with_model(text, pipeline_bsc, "bsc")
    kbio_entities = extract_entities_with_model(text, pipeline_kbio, "kbio")
    
    # Debug: mostrar algunas detecciones para los primeros archivos
    debug_mode = filename == "0000000010000000011219557 __episode_1008874064.txt.txt"
    
    if debug_mode:
        print(f"\n    ===== DEBUG {filename} =====", flush=True)
        print(f"      BSC detectó {len(bsc_entities)} entidades", flush=True)
        if bsc_entities:
            print(f"      Primeras 5 BSC: {[e.get('word', '')[:40] for e in bsc_entities[:5]]}", flush=True)
            bsc_scores = [f"{e.get('score', 0):.3f}" for e in bsc_entities[:5]]
            print(f"      BSC confianzas: {bsc_scores}", flush=True)
        print(f"      KBio detectó {len(kbio_entities)} entidades", flush=True)
        if kbio_entities:
            print(f"      Primeras 5 KBio: {[e.get('word', '')[:40] for e in kbio_entities[:5]]}", flush=True)
            kbio_scores = [f"{e.get('score', 0):.3f}" for e in kbio_entities[:5]]
            print(f"      KBio confianzas: {kbio_scores}", flush=True)
    
    # Filtrar según criterios de confianza
    accepted_entities = filter_entities_by_confidence(
        bsc_entities, kbio_entities, both_threshold, single_threshold, filename if debug_mode else None
    )
    
    if debug_mode:
        print(f"      Entidades aceptadas después de filtrar: {len(accepted_entities)}", flush=True)
        if accepted_entities:
            print(f"      Primeras 5 aceptadas: {accepted_entities[:5]}", flush=True)
        print(f"    ===== FIN DEBUG =====\n", flush=True)
    
    return {
        "filename": filename,
        "entities": accepted_entities,
        "entity_count": len(accepted_entities),
        "bsc_detections": len(bsc_entities),
        "kbio_detections": len(kbio_entities)
    }

def process_file_wrapper_models(args):
    """
    Wrapper para procesar un archivo con modelos (para ThreadPoolExecutor).
    Los modelos ya están cargados y se pasan como argumentos.
    
    Args:
        args: Tupla (file_path, pipeline_bsc, pipeline_kbio, both_threshold, single_threshold)
        
    Returns:
        Dict con los resultados
    """
    file_path, pipeline_bsc, pipeline_kbio, both_threshold, single_threshold = args
    
    return process_text_file_with_models(
        file_path, pipeline_bsc, pipeline_kbio, both_threshold, single_threshold
    )

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
    
    parser = argparse.ArgumentParser(description='Extrae entidades usando modelos BSC y KBio')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Número máximo de archivos a procesar (para pruebas)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Número de workers paralelos (por defecto: 1, los modelos son pesados)')
    parser.add_argument('--both-threshold', type=float, default=0.8,
                       help='Umbral de confianza cuando ambos modelos detectan (default: 0.8)')
    parser.add_argument('--single-threshold', type=float, default=0.98,
                       help='Umbral de confianza cuando solo un modelo detecta (default: 0.98)')
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers no está disponible. Instala con: pip install transformers torch", flush=True)
        return 1
    
    # Iniciar medición de tiempo total
    start_time_total = time.time()
    
    print("="*60, flush=True)
    print("PIPELINE DE ANONIMIZACIÓN - PASO 4: EXTRACCIÓN CON MODELOS", flush=True)
    print("="*60, flush=True)
    
    # Rutas
    # IMPORTANTE: Usamos docs/test (texto original) para detectar nuevas entidades
    # que no fueron encontradas por regex/patrones en step1 y step2
    test_dir = "docs/test"
    output_file = "pipeline_anon/entities_anon.jsonl"
    
    # Cargar resultados existentes
    print(f"\n1. Cargando resultados existentes desde {output_file}...", flush=True)
    existing_results = load_existing_results(output_file)
    print(f"  Archivos existentes: {len(existing_results)}", flush=True)
    
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
    
    # Cargar modelos una sola vez
    print(f"\n3. Cargando modelos BSC y KBio (una sola vez)...", flush=True)
    start_load_time = time.time()
    try:
        pipeline_bsc, pipeline_kbio = load_models()
        load_time = time.time() - start_load_time
        print(f"  Modelos cargados en {format_time(load_time)}", flush=True)
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los modelos: {e}", flush=True)
        return 1
    
    # Procesar cada archivo
    print(f"\n4. Procesando archivos con modelos BSC y KBio...", flush=True)
    print(f"  Total archivos a procesar: {len(txt_files)}", flush=True)
    print(f"  Umbral ambos modelos: {args.both_threshold}", flush=True)
    print(f"  Umbral modelo único: {args.single_threshold}", flush=True)
    
    # Procesar secuencialmente para evitar problemas de thread-safety con los modelos
    # Los modelos de PyTorch no son thread-safe cuando se comparten entre threads
    print(f"  Procesando secuencialmente (los modelos no son thread-safe)", flush=True)
    
    start_time = time.time()
    results = []
    total_entities_found = 0
    processed_count = 0
    
    # Filtrar archivos que ya tienen entidades de modelos (opcional: procesar solo nuevos)
    # Por ahora procesamos todos, pero podríamos filtrar si queremos
    files_to_process = txt_files
    
    print(f"  Archivos a procesar: {len(files_to_process)}", flush=True)
    
    # Procesar cada archivo secuencialmente
    for txt_file in files_to_process:
        processed_count += 1
        try:
            # Procesar directamente sin threads
            new_result = process_text_file_with_models(
                str(txt_file), pipeline_bsc, pipeline_kbio, 
                args.both_threshold, args.single_threshold
            )
            filename = new_result['filename']
            
            # Fusionar con resultados existentes si los hay
            existing = existing_results.get(filename, {})
            merged_result = merge_results(existing, new_result)
            results.append(merged_result)
            total_entities_found += merged_result["entity_count"]
            
            # Mostrar progreso cada 5 archivos o al inicio
            if processed_count % 5 == 0 or processed_count == 1:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"  Procesados {processed_count}/{len(txt_files)} archivos... "
                      f"(entidades: {total_entities_found}, tasa: {rate:.2f} archivos/s)", flush=True)
        except Exception as e:
            filename = os.path.basename(txt_file)
            print(f"  ERROR procesando {filename}: {e}", flush=True)
            print(f"    Tipo de error: {type(e).__name__}", flush=True)
            import traceback
            traceback.print_exc()
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
    
    # Actualizar resultados existentes con los nuevos resultados procesados
    print(f"\n5. Actualizando resultados en {output_file}...", flush=True)
    start_time = time.time()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Crear un diccionario con todos los resultados (existentes + nuevos procesados)
    all_results = existing_results.copy()
    
    # Contar cuántos archivos se actualizan vs cuántos son nuevos
    updated_count = 0
    new_files_count = 0
    total_new_entities_added = 0
    
    # Actualizar con los nuevos resultados procesados
    for result in results:
        filename = result["filename"]
        if filename in existing_results:
            updated_count += 1
            old_entities = set(existing_results[filename].get('entities', []))
            new_entities = set(result.get('entities', []))
            merged_entities = list(dict.fromkeys(existing_results[filename].get('entities', []) + result.get('entities', [])))
            entities_added = len(merged_entities) - len(old_entities)
            total_new_entities_added += entities_added
            
            # Mostrar comparación de entidades antes/después para algunos archivos
            if updated_count <= 5:
                old_count = existing_results[filename].get('entity_count', 0)
                new_count_entities = result.get('entity_count', 0)
                merged_count = len(merged_entities)
                print(f"  Actualizando {filename}:", flush=True)
                print(f"    Entidades existentes: {old_count}", flush=True)
                print(f"    Entidades detectadas por modelos: {new_count_entities}", flush=True)
                print(f"    Entidades después de fusionar: {merged_count} (+{entities_added})", flush=True)
                if entities_added > 0:
                    new_only = new_entities - old_entities
                    print(f"    Nuevas entidades agregadas: {list(new_only)[:5]}...", flush=True)
        else:
            new_files_count += 1
            total_new_entities_added += result.get('entity_count', 0)
        
        # Fusionar correctamente antes de guardar
        if filename in existing_results:
            merged_result = merge_results(existing_results[filename], result)
            all_results[filename] = merged_result
        else:
            all_results[filename] = result
    
    # Guardar todos los resultados (existentes + actualizados)
    with open(output_file, 'w', encoding='utf-8') as f:
        # Ordenar por nombre de archivo para mantener consistencia
        for filename in sorted(all_results.keys()):
            f.write(json.dumps(all_results[filename], ensure_ascii=False) + '\n')
    
    save_time = time.time() - start_time
    print(f"  Archivos procesados: {len(results)}", flush=True)
    print(f"  Archivos actualizados: {updated_count}", flush=True)
    print(f"  Archivos nuevos: {new_files_count}", flush=True)
    print(f"  Total nuevas entidades agregadas: {total_new_entities_added}", flush=True)
    print(f"  Total archivos guardados: {len(all_results)}", flush=True)
    print(f"  Tiempo de guardado: {format_time(save_time)}", flush=True)
    
    # Calcular tiempo total
    total_time = time.time() - start_time_total
    
    # Estadísticas
    print(f"\n6. Estadísticas:", flush=True)
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
    print(f"Procesamiento: Secuencial (modelos no son thread-safe)", flush=True)
    print(f"\nTiempos:", flush=True)
    print(f"  Procesamiento: {format_time(processing_time)}", flush=True)
    print(f"  Guardado: {format_time(save_time)}", flush=True)
    print(f"  TOTAL: {format_time(total_time)}", flush=True)
    if len(results) > 0:
        throughput = len(results) / total_time
        print(f"\nRendimiento:", flush=True)
        print(f"  Archivos por segundo: {throughput:.2f}", flush=True)
        print(f"  Entidades por segundo: {total_entities_found / total_time:.2f}", flush=True)
    print("="*60, flush=True)
    print("PASO 4 COMPLETADO", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

