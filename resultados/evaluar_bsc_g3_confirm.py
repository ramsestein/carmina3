#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador que combina BERT bsc_ehr_anon_bin con g3_confirm.

1. Usa BERT para detectar entidades en chunks (threshold 0)
2. Para cada entidad detectada, envía el chunk y la entidad a g3_confirm
3. Solo acepta entidades si g3_confirm responde "Sí"
4. Evalúa sobre 50 ejemplos aleatorios del dataset de train
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
    from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers no está disponible", flush=True)
    sys.exit(1)

# Importar PEFT para cargar modelo LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("ERROR: peft no está disponible", flush=True)
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

def load_bert_model():
    """
    Carga el modelo BERT bsc_ehr_anon_bin.
    
    Returns:
        Pipeline del modelo BERT
    """
    print("Cargando modelo BERT bsc_ehr_anon_bin...", flush=True)
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    
    bsc_model_path = "models/bsc_ehr_anon_bin"
    
    try:
        bsc_tokenizer = AutoTokenizer.from_pretrained(bsc_model_path)
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
        print("  Modelo BERT cargado exitosamente", flush=True)
        return pipeline_bsc
    except Exception as e:
        print(f"  ERROR cargando modelo BERT: {e}", flush=True)
        raise e

def load_g3_confirm_model():
    """
    Carga el modelo g3_confirm (Gemma con LoRA).
    
    Returns:
        Tuple (model, tokenizer)
    """
    print("Cargando modelo g3_confirm...", flush=True)
    
    base_model_name = "google/gemma-3-270m"
    adapter_path = "models/g3_confirm"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}", flush=True)
    
    try:
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Cargar modelo base
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Cargar adaptador LoRA
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()  # Modo evaluación
        
        print("  Modelo g3_confirm cargado exitosamente", flush=True)
        return model, tokenizer
    except Exception as e:
        print(f"  ERROR cargando modelo g3_confirm: {e}", flush=True)
        raise e

def format_prompt_for_g3(context: str, text: str) -> str:
    """
    Formatea un prompt para g3_confirm (formato clasificación).
    
    NOTA: Este formato replica la estructura del input de lora_dataset.jsonl:
    {"input": {"context": "...", "text": "..."}, "output": "Sí"/"No"}
    Pero solo enviamos el input (sin el output), ya que el output es solo para entrenamiento.
    
    Args:
        context: Contexto del texto (chunk alrededor de la entidad)
        text: Texto a clasificar (entidad detectada)
        
    Returns:
        String con el prompt formateado (solo input, sin output)
    """
    prompt = f"""<start_of_turn>user
Eres un asistente experto en identificar entidades médicas en textos españoles que deben ser anonimizadas.

Contexto: {context}

Pregunta: ¿Es "{text}" una entidad médica que debe ser anonimizada?<end_of_turn>
<start_of_turn>model
"""
    return prompt

def query_g3_confirm(model, tokenizer, prompt: str, max_new_tokens: int = 10, temperature: float = 0.8) -> str:
    """
    Consulta el modelo g3_confirm con un prompt.
    
    Args:
        model: Modelo g3_confirm
        tokenizer: Tokenizador
        prompt: Prompt formateado
        max_new_tokens: Máximo número de tokens a generar
        temperature: Temperatura para el muestreo (default: 0.3)
        
    Returns:
        Respuesta del modelo ("Sí" o "No")
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar solo los nuevos tokens
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Normalizar respuesta
        generated_text = generated_text.strip().lower()
        if "sí" in generated_text or "si" in generated_text or "yes" in generated_text:
            return "Sí"
        else:
            return "No"
    except Exception as e:
        print(f"    ERROR consultando g3_confirm: {e}", flush=True)
        return "No"

def extract_entities_with_bert(text: str, pipeline_model, threshold: float = 0.0) -> List[Dict]:
    """
    Extrae entidades de un texto usando el modelo BERT con un threshold.
    
    Args:
        text: Texto a procesar
        pipeline_model: Pipeline del modelo BERT
        threshold: Threshold de confianza (0 = aceptar todas)
        
    Returns:
        Lista de diccionarios con 'word' (texto) y 'start' (posición)
    """
    entities = []
    
    try:
        max_chunk_length = 450
        
        if len(text) <= max_chunk_length:
            model_entities = pipeline_model(text)
            
            if isinstance(model_entities, list):
                for entity in model_entities:
                    score = entity.get('score', 0.0)
                    if score >= threshold:
                        entity_text = entity.get('word', '').strip()
                        entity_start = entity.get('start', 0)
                        if entity_text:
                            entities.append({
                                'word': entity_text,
                                'start': entity_start,
                                'end': entity.get('end', entity_start + len(entity_text))
                            })
        else:
            # Texto largo, dividir en chunks
            chunk_size = max_chunk_length
            overlap_size = 100
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
                            score = entity.get('score', 0.0)
                            if score >= threshold:
                                entity_text = entity.get('word', '').strip()
                                
                                if not entity_text:
                                    continue
                                
                                entity_start_relative = entity.get('start', 0)
                                entity_start_absolute = i + entity_start_relative
                                entity_end_absolute = i + entity.get('end', entity_start_relative + len(entity_text))
                                entity_text_normalized = entity_text.lower().strip()
                                position_key = (entity_start_absolute // 10) * 10
                                entity_key = (entity_text_normalized, position_key)
                                
                                if entity_key not in seen_entities:
                                    entities.append({
                                        'word': entity_text,
                                        'start': entity_start_absolute,
                                        'end': entity_end_absolute
                                    })
                                    seen_entities[entity_key] = {
                                        'text': entity_text,
                                        'position': entity_start_absolute,
                                        'score': score
                                    }
                                else:
                                    existing_score = seen_entities[entity_key].get('score', 0.0)
                                    if score > existing_score:
                                        seen_entities[entity_key]['score'] = score
                                        seen_entities[entity_key]['text'] = entity_text
                                        for idx in range(len(entities) - 1, -1, -1):
                                            if entities[idx]['word'].lower().strip() == entity_text_normalized:
                                                entities[idx] = {
                                                    'word': entity_text,
                                                    'start': entity_start_absolute,
                                                    'end': entity_end_absolute
                                                }
                                                break
                
                except Exception as e:
                    pass
                
                i += chunk_size - overlap_size
                if i >= len(text):
                    break
        
        return entities
    
    except Exception as e:
        print(f"  ERROR extrayendo entidades con BERT: {e}", flush=True)
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

def extract_context_around_entity(text: str, entity_start: int, entity_end: int, context_size: int = 200) -> str:
    """
    Extrae el contexto alrededor de una entidad en el texto.
    
    Args:
        text: Texto completo
        entity_start: Posición de inicio de la entidad
        entity_end: Posición de fin de la entidad
        context_size: Tamaño del contexto a extraer antes y después (en caracteres)
        
    Returns:
        Contexto alrededor de la entidad
    """
    # Calcular inicio y fin del contexto
    context_start = max(0, entity_start - context_size)
    context_end = min(len(text), entity_end + context_size)
    
    return text[context_start:context_end]

def calculate_strict_metrics(predicted: List[str], reference: List[str]) -> Dict:
    """
    Calcula métricas strict (coincidencia exacta, case-insensitive).
    
    Args:
        predicted: Lista de entidades predichas
        reference: Lista de entidades de referencia
        
    Returns:
        Dict con métricas
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
    
    Args:
        predicted: Lista de entidades predichas
        reference: Lista de entidades de referencia
        
    Returns:
        Dict con métricas
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Normalizar listas
    pred_normalized = [e.strip().lower() for e in predicted if e.strip()]
    ref_normalized = [e.strip().lower() for e in reference if e.strip()]
    
    # Marcar entidades de referencia que fueron encontradas
    ref_matched = [False] * len(ref_normalized)
    
    # Para cada predicción, buscar si coincide con alguna referencia (relaxed)
    for pred_entity in pred_normalized:
        matched = False
        for idx, ref_entity in enumerate(ref_normalized):
            if is_entity_included(pred_entity, ref_entity) or is_entity_included(ref_entity, pred_entity):
                if not ref_matched[idx]:
                    true_positives += 1
                    ref_matched[idx] = True
                    matched = True
                    break
        
        if not matched:
            false_positives += 1
    
    # Contar false negatives
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
    print("="*60, flush=True)
    print("EVALUACIÓN BERT + G3_CONFIRM", flush=True)
    print("="*60, flush=True)
    
    # Rutas
    train_dir = "docs/train"
    reference_file = "docs/train/entities_train.jsonl"
    output_dir = "resultados"
    output_file = os.path.join(output_dir, "metricas_bsc_g3_confirm.json")
    
    # Crear directorio de resultados si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar que existe el directorio de train
    if not os.path.exists(train_dir):
        print(f"ERROR: No existe el directorio {train_dir}", flush=True)
        sys.exit(1)
    
    # Verificar que existe el archivo de entidades
    if not os.path.exists(reference_file):
        print(f"ERROR: No existe el archivo {reference_file}", flush=True)
        sys.exit(1)
    
    # Cargar modelos
    start_time = time.time()
    pipeline_bert = load_bert_model()
    bert_load_time = time.time() - start_time
    
    start_time = time.time()
    model_g3, tokenizer_g3 = load_g3_confirm_model()
    g3_load_time = time.time() - start_time
    
    print(f"\nTiempo de carga:")
    print(f"  BERT: {bert_load_time:.2f} segundos", flush=True)
    print(f"  G3_confirm: {g3_load_time:.2f} segundos", flush=True)
    
    # Cargar entidades de referencia
    reference_entities = load_reference_entities(reference_file)
    
    # Obtener lista de archivos .txt en docs/train
    txt_files = list(Path(train_dir).glob("*.txt"))
    total_files_available = len(txt_files)
    
    # Seleccionar 50 archivos aleatorios
    random.seed(42)
    files_to_process = random.sample(txt_files, min(50, total_files_available))
    
    print(f"\nArchivos disponibles: {total_files_available}", flush=True)
    print(f"Archivos a procesar (aleatorios): {len(files_to_process)}", flush=True)
    
    # Procesar ejemplos
    print("\nProcesando ejemplos...", flush=True)
    
    results = []
    
    # Métricas BERT solo
    bert_only_strict_tp = 0
    bert_only_strict_fp = 0
    bert_only_strict_fn = 0
    bert_only_relaxed_tp = 0
    bert_only_relaxed_fp = 0
    bert_only_relaxed_fn = 0
    
    # Métricas BERT + G3
    bert_g3_strict_tp = 0
    bert_g3_strict_fp = 0
    bert_g3_strict_fn = 0
    bert_g3_relaxed_tp = 0
    bert_g3_relaxed_fp = 0
    bert_g3_relaxed_fn = 0
    
    # Estadísticas de tiempo
    total_bert_time = 0.0
    total_g3_time = 0.0
    total_metrics_time = 0.0
    bert_times = []
    g3_times = []
    g3_queries_count = 0
    
    processing_start = time.time()
    
    for idx, txt_file in enumerate(files_to_process, 1):
        example_start = time.time()
        filename = txt_file.name
        filename_normalized = filename.replace('.txt.txt', '.txt')
        
        # Leer texto completo
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                full_text = f.read()
        except Exception as e:
            print(f"  ERROR leyendo {filename}: {e}", flush=True)
            continue
        
        # Obtener entidades de referencia
        reference_entities_list = reference_entities.get(filename_normalized, [])
        if not reference_entities_list:
            reference_entities_list = reference_entities.get(filename, [])
        
        # Extraer entidades con BERT (threshold 0)
        bert_start = time.time()
        bert_entities_data = extract_entities_with_bert(full_text, pipeline_bert, threshold=0.0)
        bert_time = time.time() - bert_start
        total_bert_time += bert_time
        bert_times.append(bert_time)
        
        # Obtener lista de textos de entidades BERT
        bert_entities_text = [e['word'] for e in bert_entities_data]
        
        # Calcular métricas BERT solo
        metrics_start = time.time()
        bert_only_strict = calculate_strict_metrics(bert_entities_text, reference_entities_list)
        bert_only_relaxed = calculate_relaxed_metrics(bert_entities_text, reference_entities_list)
        
        bert_only_strict_tp += bert_only_strict['true_positives']
        bert_only_strict_fp += bert_only_strict['false_positives']
        bert_only_strict_fn += bert_only_strict['false_negatives']
        bert_only_relaxed_tp += bert_only_relaxed['true_positives']
        bert_only_relaxed_fp += bert_only_relaxed['false_positives']
        bert_only_relaxed_fn += bert_only_relaxed['false_negatives']
        
        # Filtrar entidades con g3_confirm
        confirmed_entities = []
        g3_start = time.time()
        for entity_data in bert_entities_data:
            entity_text = entity_data['word']
            entity_start = entity_data['start']
            entity_end = entity_data['end']
            
            # Extraer contexto alrededor de la entidad
            context = extract_context_around_entity(full_text, entity_start, entity_end, context_size=200)
            
            # Formatear prompt como en lora_dataset.jsonl
            prompt = format_prompt_for_g3(context, entity_text)
            
            query_start = time.time()
            response = query_g3_confirm(model_g3, tokenizer_g3, prompt)
            query_time = time.time() - query_start
            g3_times.append(query_time)
            g3_queries_count += 1
            
            if response == "Sí":
                confirmed_entities.append(entity_text)
        g3_time = time.time() - g3_start
        total_g3_time += g3_time
        
        # Calcular métricas BERT + G3
        bert_g3_strict = calculate_strict_metrics(confirmed_entities, reference_entities_list)
        bert_g3_relaxed = calculate_relaxed_metrics(confirmed_entities, reference_entities_list)
        metrics_time = time.time() - metrics_start
        total_metrics_time += metrics_time
        
        bert_g3_strict_tp += bert_g3_strict['true_positives']
        bert_g3_strict_fp += bert_g3_strict['false_positives']
        bert_g3_strict_fn += bert_g3_strict['false_negatives']
        bert_g3_relaxed_tp += bert_g3_relaxed['true_positives']
        bert_g3_relaxed_fp += bert_g3_relaxed['false_positives']
        bert_g3_relaxed_fn += bert_g3_relaxed['false_negatives']
        
        example_time = time.time() - example_start
        
        results.append({
            'filename': filename,
            'bert_entities_count': len(bert_entities_data),
            'confirmed_entities_count': len(confirmed_entities),
            'reference_entities_count': len(reference_entities_list),
            'g3_queries_count': len(bert_entities_data),
            'times': {
                'bert_extraction': bert_time,
                'g3_confirmation': g3_time,
                'metrics_calculation': metrics_time,
                'total': example_time
            },
            'bert_only': {
                'strict': bert_only_strict,
                'relaxed': bert_only_relaxed
            },
            'bert_g3': {
                'strict': bert_g3_strict,
                'relaxed': bert_g3_relaxed
            }
        })
        
        if idx % 10 == 0:
            avg_bert = total_bert_time / idx
            avg_g3 = total_g3_time / idx
            avg_example = (time.time() - processing_start) / idx
            
            # Calcular métricas acumuladas hasta ahora (BERT + G3)
            temp_bert_g3_strict_precision = bert_g3_strict_tp / (bert_g3_strict_tp + bert_g3_strict_fp) if (bert_g3_strict_tp + bert_g3_strict_fp) > 0 else 0.0
            temp_bert_g3_strict_recall = bert_g3_strict_tp / (bert_g3_strict_tp + bert_g3_strict_fn) if (bert_g3_strict_tp + bert_g3_strict_fn) > 0 else 0.0
            temp_bert_g3_strict_f1 = 2 * (temp_bert_g3_strict_precision * temp_bert_g3_strict_recall) / (temp_bert_g3_strict_precision + temp_bert_g3_strict_recall) if (temp_bert_g3_strict_precision + temp_bert_g3_strict_recall) > 0 else 0.0
            
            temp_bert_g3_relaxed_precision = bert_g3_relaxed_tp / (bert_g3_relaxed_tp + bert_g3_relaxed_fp) if (bert_g3_relaxed_tp + bert_g3_relaxed_fp) > 0 else 0.0
            temp_bert_g3_relaxed_recall = bert_g3_relaxed_tp / (bert_g3_relaxed_tp + bert_g3_relaxed_fn) if (bert_g3_relaxed_tp + bert_g3_relaxed_fn) > 0 else 0.0
            temp_bert_g3_relaxed_f1 = 2 * (temp_bert_g3_relaxed_precision * temp_bert_g3_relaxed_recall) / (temp_bert_g3_relaxed_precision + temp_bert_g3_relaxed_recall) if (temp_bert_g3_relaxed_precision + temp_bert_g3_relaxed_recall) > 0 else 0.0
            
            print(f"  Procesados {idx}/{len(files_to_process)} archivos", flush=True)
            print(f"    Tiempo promedio por archivo: {avg_example:.2f}s", flush=True)
            print(f"    Tiempo promedio BERT: {avg_bert:.3f}s, G3: {avg_g3:.3f}s", flush=True)
            print(f"    Consultas G3 totales: {g3_queries_count} (promedio: {g3_queries_count/idx:.1f} por archivo)", flush=True)
            print(f"    BERT+G3 (strict): P={temp_bert_g3_strict_precision:.4f}, R={temp_bert_g3_strict_recall:.4f}, F1={temp_bert_g3_strict_f1:.4f}", flush=True)
            print(f"    BERT+G3 (relaxed): P={temp_bert_g3_relaxed_precision:.4f}, R={temp_bert_g3_relaxed_recall:.4f}, F1={temp_bert_g3_relaxed_f1:.4f}", flush=True)
    
    processing_time = time.time() - processing_start
    
    # Calcular métricas globales BERT solo
    bert_only_strict_precision = bert_only_strict_tp / (bert_only_strict_tp + bert_only_strict_fp) if (bert_only_strict_tp + bert_only_strict_fp) > 0 else 0.0
    bert_only_strict_recall = bert_only_strict_tp / (bert_only_strict_tp + bert_only_strict_fn) if (bert_only_strict_tp + bert_only_strict_fn) > 0 else 0.0
    bert_only_strict_f1 = 2 * (bert_only_strict_precision * bert_only_strict_recall) / (bert_only_strict_precision + bert_only_strict_recall) if (bert_only_strict_precision + bert_only_strict_recall) > 0 else 0.0
    
    bert_only_relaxed_precision = bert_only_relaxed_tp / (bert_only_relaxed_tp + bert_only_relaxed_fp) if (bert_only_relaxed_tp + bert_only_relaxed_fp) > 0 else 0.0
    bert_only_relaxed_recall = bert_only_relaxed_tp / (bert_only_relaxed_tp + bert_only_relaxed_fn) if (bert_only_relaxed_tp + bert_only_relaxed_fn) > 0 else 0.0
    bert_only_relaxed_f1 = 2 * (bert_only_relaxed_precision * bert_only_relaxed_recall) / (bert_only_relaxed_precision + bert_only_relaxed_recall) if (bert_only_relaxed_precision + bert_only_relaxed_recall) > 0 else 0.0
    
    # Calcular métricas globales BERT + G3
    bert_g3_strict_precision = bert_g3_strict_tp / (bert_g3_strict_tp + bert_g3_strict_fp) if (bert_g3_strict_tp + bert_g3_strict_fp) > 0 else 0.0
    bert_g3_strict_recall = bert_g3_strict_tp / (bert_g3_strict_tp + bert_g3_strict_fn) if (bert_g3_strict_tp + bert_g3_strict_fn) > 0 else 0.0
    bert_g3_strict_f1 = 2 * (bert_g3_strict_precision * bert_g3_strict_recall) / (bert_g3_strict_precision + bert_g3_strict_recall) if (bert_g3_strict_precision + bert_g3_strict_recall) > 0 else 0.0
    
    bert_g3_relaxed_precision = bert_g3_relaxed_tp / (bert_g3_relaxed_tp + bert_g3_relaxed_fp) if (bert_g3_relaxed_tp + bert_g3_relaxed_fp) > 0 else 0.0
    bert_g3_relaxed_recall = bert_g3_relaxed_tp / (bert_g3_relaxed_tp + bert_g3_relaxed_fn) if (bert_g3_relaxed_tp + bert_g3_relaxed_fn) > 0 else 0.0
    bert_g3_relaxed_f1 = 2 * (bert_g3_relaxed_precision * bert_g3_relaxed_recall) / (bert_g3_relaxed_precision + bert_g3_relaxed_recall) if (bert_g3_relaxed_precision + bert_g3_relaxed_recall) > 0 else 0.0
    
    # Calcular estadísticas de tiempo
    avg_bert_time = total_bert_time / len(files_to_process) if files_to_process else 0
    avg_g3_time = total_g3_time / len(files_to_process) if files_to_process else 0
    avg_metrics_time = total_metrics_time / len(files_to_process) if files_to_process else 0
    avg_example_time = processing_time / len(files_to_process) if files_to_process else 0
    avg_g3_query_time = sum(g3_times) / len(g3_times) if g3_times else 0
    
    # Guardar resultados
    output_data = {
        'model': 'bsc_ehr_anon_bin + g3_confirm',
        'train_directory': train_dir,
        'reference_file': reference_file,
        'total_files_available': total_files_available,
        'total_files_processed': len(files_to_process),
        'files_selection': 'random_50',
        'bert_threshold': 0.0,
        'context_size': 200,
        'timing': {
            'model_loading': {
                'bert_load_time_seconds': bert_load_time,
                'g3_load_time_seconds': g3_load_time,
                'total_load_time_seconds': bert_load_time + g3_load_time
            },
            'processing': {
                'total_processing_time_seconds': processing_time,
                'total_bert_time_seconds': total_bert_time,
                'total_g3_time_seconds': total_g3_time,
                'total_metrics_time_seconds': total_metrics_time,
                'average_per_example': {
                    'bert_extraction_seconds': avg_bert_time,
                    'g3_confirmation_seconds': avg_g3_time,
                    'metrics_calculation_seconds': avg_metrics_time,
                    'total_seconds': avg_example_time
                },
                'g3_queries': {
                    'total_queries': g3_queries_count,
                    'average_queries_per_file': g3_queries_count / len(files_to_process) if files_to_process else 0,
                    'average_time_per_query_seconds': avg_g3_query_time,
                    'min_query_time_seconds': min(g3_times) if g3_times else 0,
                    'max_query_time_seconds': max(g3_times) if g3_times else 0
                }
            },
            'total_time_seconds': bert_load_time + g3_load_time + processing_time
        },
        'metrics': {
            'bert_only': {
                'strict': {
                    'precision': bert_only_strict_precision,
                    'recall': bert_only_strict_recall,
                    'f1': bert_only_strict_f1,
                    'true_positives': bert_only_strict_tp,
                    'false_positives': bert_only_strict_fp,
                    'false_negatives': bert_only_strict_fn
                },
                'relaxed': {
                    'precision': bert_only_relaxed_precision,
                    'recall': bert_only_relaxed_recall,
                    'f1': bert_only_relaxed_f1,
                    'true_positives': bert_only_relaxed_tp,
                    'false_positives': bert_only_relaxed_fp,
                    'false_negatives': bert_only_relaxed_fn
                }
            },
            'bert_g3': {
                'strict': {
                    'precision': bert_g3_strict_precision,
                    'recall': bert_g3_strict_recall,
                    'f1': bert_g3_strict_f1,
                    'true_positives': bert_g3_strict_tp,
                    'false_positives': bert_g3_strict_fp,
                    'false_negatives': bert_g3_strict_fn
                },
                'relaxed': {
                    'precision': bert_g3_relaxed_precision,
                    'recall': bert_g3_relaxed_recall,
                    'f1': bert_g3_relaxed_f1,
                    'true_positives': bert_g3_relaxed_tp,
                    'false_positives': bert_g3_relaxed_fp,
                    'false_negatives': bert_g3_relaxed_fn
                }
            }
        },
        'per_example_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    print("\n" + "="*60, flush=True)
    print("RESULTADOS", flush=True)
    print("="*60, flush=True)
    
    print(f"\n⏱️  TIEMPOS:", flush=True)
    print(f"  Carga de modelos:", flush=True)
    print(f"    BERT: {bert_load_time:.2f}s", flush=True)
    print(f"    G3_confirm: {g3_load_time:.2f}s", flush=True)
    print(f"    Total carga: {bert_load_time + g3_load_time:.2f}s", flush=True)
    
    print(f"\n  Procesamiento:", flush=True)
    print(f"    Total: {processing_time:.2f}s", flush=True)
    print(f"    Por archivo (promedio): {avg_example_time:.2f}s", flush=True)
    print(f"    - Extracción BERT: {avg_bert_time:.3f}s por archivo ({total_bert_time:.2f}s total)", flush=True)
    print(f"    - Confirmación G3: {avg_g3_time:.3f}s por archivo ({total_g3_time:.2f}s total)", flush=True)
    print(f"    - Cálculo métricas: {avg_metrics_time:.4f}s por archivo ({total_metrics_time:.2f}s total)", flush=True)
    
    print(f"\n  Consultas G3:", flush=True)
    print(f"    Total consultas: {g3_queries_count}", flush=True)
    print(f"    Promedio por archivo: {g3_queries_count / len(files_to_process):.1f}", flush=True)
    print(f"    Tiempo promedio por consulta: {avg_g3_query_time:.3f}s", flush=True)
    if g3_times:
        print(f"    Tiempo mínimo: {min(g3_times):.3f}s, máximo: {max(g3_times):.3f}s", flush=True)
    
    print(f"\n  Tiempo total: {output_data['timing']['total_time_seconds']:.2f}s", flush=True)
    
    # Calcular métricas adicionales
    bert_only_strict_total = bert_only_strict_tp + bert_only_strict_fp + bert_only_strict_fn
    bert_only_relaxed_total = bert_only_relaxed_tp + bert_only_relaxed_fp + bert_only_relaxed_fn
    bert_g3_strict_total = bert_g3_strict_tp + bert_g3_strict_fp + bert_g3_strict_fn
    bert_g3_relaxed_total = bert_g3_relaxed_tp + bert_g3_relaxed_fp + bert_g3_relaxed_fn
    
    print(f"\n{'='*60}", flush=True)
    print(f"📊 MÉTRICAS BERT SOLO (sin filtro G3)", flush=True)
    print(f"{'='*60}", flush=True)
    
    print(f"\n  STRICT (Coincidencia exacta):", flush=True)
    print(f"    Precision: {bert_only_strict_precision:.4f} ({bert_only_strict_precision*100:.2f}%)", flush=True)
    print(f"    Recall:    {bert_only_strict_recall:.4f} ({bert_only_strict_recall*100:.2f}%)", flush=True)
    print(f"    F1-Score:  {bert_only_strict_f1:.4f} ({bert_only_strict_f1*100:.2f}%)", flush=True)
    print(f"    True Positives:  {bert_only_strict_tp}", flush=True)
    print(f"    False Positives: {bert_only_strict_fp}", flush=True)
    print(f"    False Negatives: {bert_only_strict_fn}", flush=True)
    print(f"    Total: {bert_only_strict_total}", flush=True)
    
    print(f"\n  RELAXED (Inclusión parcial):", flush=True)
    print(f"    Precision: {bert_only_relaxed_precision:.4f} ({bert_only_relaxed_precision*100:.2f}%)", flush=True)
    print(f"    Recall:    {bert_only_relaxed_recall:.4f} ({bert_only_relaxed_recall*100:.2f}%)", flush=True)
    print(f"    F1-Score:  {bert_only_relaxed_f1:.4f} ({bert_only_relaxed_f1*100:.2f}%)", flush=True)
    print(f"    True Positives:  {bert_only_relaxed_tp}", flush=True)
    print(f"    False Positives: {bert_only_relaxed_fp}", flush=True)
    print(f"    False Negatives: {bert_only_relaxed_fn}", flush=True)
    print(f"    Total: {bert_only_relaxed_total}", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"📊 MÉTRICAS BERT + G3_CONFIRM (con filtro G3)", flush=True)
    print(f"{'='*60}", flush=True)
    
    print(f"\n  STRICT (Coincidencia exacta):", flush=True)
    print(f"    Precision: {bert_g3_strict_precision:.4f} ({bert_g3_strict_precision*100:.2f}%)", flush=True)
    print(f"    Recall:    {bert_g3_strict_recall:.4f} ({bert_g3_strict_recall*100:.2f}%)", flush=True)
    print(f"    F1-Score:  {bert_g3_strict_f1:.4f} ({bert_g3_strict_f1*100:.2f}%)", flush=True)
    print(f"    True Positives:  {bert_g3_strict_tp}", flush=True)
    print(f"    False Positives: {bert_g3_strict_fp}", flush=True)
    print(f"    False Negatives: {bert_g3_strict_fn}", flush=True)
    print(f"    Total: {bert_g3_strict_total}", flush=True)
    
    print(f"\n  RELAXED (Inclusión parcial):", flush=True)
    print(f"    Precision: {bert_g3_relaxed_precision:.4f} ({bert_g3_relaxed_precision*100:.2f}%)", flush=True)
    print(f"    Recall:    {bert_g3_relaxed_recall:.4f} ({bert_g3_relaxed_recall*100:.2f}%)", flush=True)
    print(f"    F1-Score:  {bert_g3_relaxed_f1:.4f} ({bert_g3_relaxed_f1*100:.2f}%)", flush=True)
    print(f"    True Positives:  {bert_g3_relaxed_tp}", flush=True)
    print(f"    False Positives: {bert_g3_relaxed_fp}", flush=True)
    print(f"    False Negatives: {bert_g3_relaxed_fn}", flush=True)
    print(f"    Total: {bert_g3_relaxed_total}", flush=True)
    
    # Comparación
    print(f"\n{'='*60}", flush=True)
    print(f"📈 COMPARACIÓN BERT vs BERT+G3", flush=True)
    print(f"{'='*60}", flush=True)
    
    print(f"\n  STRICT:", flush=True)
    print(f"    Precision: {bert_only_strict_precision:.4f} → {bert_g3_strict_precision:.4f} ({'+' if bert_g3_strict_precision > bert_only_strict_precision else ''}{(bert_g3_strict_precision - bert_only_strict_precision)*100:.2f}%)", flush=True)
    print(f"    Recall:    {bert_only_strict_recall:.4f} → {bert_g3_strict_recall:.4f} ({'+' if bert_g3_strict_recall > bert_only_strict_recall else ''}{(bert_g3_strict_recall - bert_only_strict_recall)*100:.2f}%)", flush=True)
    print(f"    F1-Score:  {bert_only_strict_f1:.4f} → {bert_g3_strict_f1:.4f} ({'+' if bert_g3_strict_f1 > bert_only_strict_f1 else ''}{(bert_g3_strict_f1 - bert_only_strict_f1)*100:.2f}%)", flush=True)
    
    print(f"\n  RELAXED:", flush=True)
    print(f"    Precision: {bert_only_relaxed_precision:.4f} → {bert_g3_relaxed_precision:.4f} ({'+' if bert_g3_relaxed_precision > bert_only_relaxed_precision else ''}{(bert_g3_relaxed_precision - bert_only_relaxed_precision)*100:.2f}%)", flush=True)
    print(f"    Recall:    {bert_only_relaxed_recall:.4f} → {bert_g3_relaxed_recall:.4f} ({'+' if bert_g3_relaxed_recall > bert_only_relaxed_recall else ''}{(bert_g3_relaxed_recall - bert_only_relaxed_recall)*100:.2f}%)", flush=True)
    print(f"    F1-Score:  {bert_g3_relaxed_f1:.4f} → {bert_g3_relaxed_f1:.4f} ({'+' if bert_g3_relaxed_f1 > bert_only_relaxed_f1 else ''}{(bert_g3_relaxed_f1 - bert_only_relaxed_f1)*100:.2f}%)", flush=True)
    
    # Calcular estadísticas adicionales
    total_bert_entities = sum(r['bert_entities_count'] for r in results)
    total_confirmed = sum(r['confirmed_entities_count'] for r in results)
    total_reference = sum(r['reference_entities_count'] for r in results)
    confirmation_rate = (total_confirmed / total_bert_entities * 100) if total_bert_entities > 0 else 0.0
    
    # Calcular cuántas entidades válidas fueron rechazadas por G3
    # TP rechazadas = TP de BERT - TP de BERT+G3
    bert_only_tp_rejected = bert_only_strict_tp - bert_g3_strict_tp
    bert_only_fp_rejected = bert_only_strict_fp - bert_g3_strict_fp
    rejection_rate_tp = (bert_only_tp_rejected / bert_only_strict_tp * 100) if bert_only_strict_tp > 0 else 0.0
    rejection_rate_fp = (bert_only_fp_rejected / bert_only_strict_fp * 100) if bert_only_strict_fp > 0 else 0.0
    
    print(f"\n📈 ESTADÍSTICAS ADICIONALES:", flush=True)
    print(f"  Entidades detectadas por BERT: {total_bert_entities}", flush=True)
    print(f"  Entidades confirmadas por G3: {total_confirmed} ({confirmation_rate:.2f}%)", flush=True)
    print(f"  Entidades de referencia: {total_reference}", flush=True)
    print(f"  Tasa de confirmación G3: {confirmation_rate:.2f}%", flush=True)
    
    print(f"\n📉 ANÁLISIS DE FILTRADO G3:", flush=True)
    print(f"  True Positives rechazados por G3: {bert_only_tp_rejected} ({rejection_rate_tp:.2f}% de los TP de BERT)", flush=True)
    print(f"  False Positives rechazados por G3: {bert_only_fp_rejected} ({rejection_rate_fp:.2f}% de los FP de BERT)", flush=True)
    print(f"  Balance: G3 rechazó {bert_only_tp_rejected} TP válidos vs {bert_only_fp_rejected} FP incorrectos", flush=True)
    if bert_only_tp_rejected > bert_only_fp_rejected:
        print(f"  ⚠️  ADVERTENCIA: G3 está rechazando más entidades válidas que falsos positivos", flush=True)
    else:
        print(f"  ✓ G3 está filtrando más falsos positivos que entidades válidas", flush=True)
    
    print(f"\n💾 Resultados guardados en: {output_file}", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

