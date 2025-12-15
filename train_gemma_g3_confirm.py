#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para fine-tunear el modelo Gemma 3 270M usando LoRA con el dataset lora_dataset.jsonl.
El modelo fine-tuneado se guardará como g3_confirm.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Importar torch primero
import torch

# Importar datasets
from datasets import Dataset

# Importar transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Importar peft después de transformers
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("ERROR: peft no está instalado. Instálalo con: pip install peft", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"ERROR al importar peft: {e}", flush=True)
    print("Intenta actualizar: pip install --upgrade peft transformers", flush=True)
    sys.exit(1)

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def load_dataset(jsonl_file: str) -> List[Dict]:
    """
    Carga el dataset desde un archivo JSONL.
    
    Args:
        jsonl_file: Ruta al archivo JSONL
        
    Returns:
        Lista de diccionarios con los datos
    """
    data = []
    print(f"Cargando dataset desde {jsonl_file}...", flush=True)
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"  ADVERTENCIA: Error parseando línea {line_num}: {e}", flush=True)
                    continue
    
    print(f"  Ejemplos cargados: {len(data)}", flush=True)
    return data

def format_prompt(context: str, text: str, output: str = None) -> str:
    """
    Formatea un ejemplo como prompt para Gemma (formato clasificación).
    
    Args:
        context: Contexto del texto
        text: Texto a clasificar
        output: Respuesta esperada ("Sí" o "No"), None para inferencia
        
    Returns:
        String con el prompt formateado
    """
    # Formato de prompt para Gemma (modelo instruct)
    prompt = f"""<start_of_turn>user
Eres un asistente experto en identificar entidades médicas en textos españoles que deben ser anonimizadas.

Contexto: {context}

Pregunta: ¿Es "{text}" una entidad médica que debe ser anonimizada?<end_of_turn>
<start_of_turn>model
"""
    
    if output is not None:
        prompt += f"{output}<end_of_turn>"
    
    return prompt

def format_prompt_chunks(chunk_text: str, entities: List[str]) -> str:
    """
    Formatea un ejemplo de chunks como prompt para Gemma (formato extracción).
    
    Args:
        chunk_text: Chunk de texto
        entities: Lista de entidades encontradas en el chunk
        
    Returns:
        String con el prompt formateado
    """
    # Formatear lista de entidades como string
    if entities:
        entities_str = ", ".join(entities)
    else:
        entities_str = "ninguna"
    
    # Formato de prompt para Gemma (modelo instruct)
    prompt = f"""<start_of_turn>user
Eres un asistente experto en extraer entidades médicas de textos españoles que deben ser anonimizadas.

Texto: {chunk_text}

¿Qué entidades médicas contiene este texto que deben ser anonimizadas?<end_of_turn>
<start_of_turn>model
[{entities_str}]<end_of_turn>
"""
    
    return prompt

def prepare_dataset(data: List[Dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Prepara el dataset para el entrenamiento.
    Soporta dos formatos:
    1. Formato clasificación: {"input": {"context": "...", "text": "..."}, "output": "Sí"/"No"}
    2. Formato chunks: {"input": "chunk de texto", "output": ["entidad1", "entidad2", ...]}
    
    Args:
        data: Lista de ejemplos
        tokenizer: Tokenizador del modelo
        max_length: Longitud máxima de secuencia
        
    Returns:
        Dataset preparado
    """
    print("Preparando dataset...", flush=True)
    
    texts = []
    
    for example in data:
        input_data = example.get('input', {})
        output = example.get('output', '')
        
        # Detectar formato: si input es string, es formato chunks
        if isinstance(input_data, str):
            # Formato chunks: {"input": "chunk", "output": ["entidad1", ...]}
            chunk_text = input_data
            entities = output if isinstance(output, list) else []
            prompt = format_prompt_chunks(chunk_text, entities)
        else:
            # Formato clasificación: {"input": {"context": "...", "text": "..."}, "output": "Sí"/"No"}
            context = input_data.get('context', '')
            text = input_data.get('text', '')
            prompt = format_prompt(context, text, output)
        
        texts.append(prompt)
    
    # Tokenizar
    print("Tokenizando textos...", flush=True)
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None  # Devolver listas, no tensores
    )
    
    # Para causal LM, los labels son iguales a input_ids
    # Pero necesitamos ignorar los tokens del prompt (solo calcular loss en la respuesta)
    labels = []
    for i, input_ids in enumerate(tokenized['input_ids']):
        # Crear labels: -100 para ignorar, token_id para calcular loss
        label = [-100] * len(input_ids)
        # Encontrar dónde empieza la respuesta del modelo
        prompt_text = texts[i]
        model_start = prompt_text.find("<start_of_turn>model\n")
        if model_start != -1:
            # Tokenizar solo la parte del prompt hasta el inicio de la respuesta
            prompt_part = prompt_text[:model_start + len("<start_of_turn>model\n")]
            prompt_tokens = tokenizer.encode(prompt_part, add_special_tokens=False)
            # Los tokens después del prompt son los que queremos predecir
            answer_start_idx = len(prompt_tokens)
            label[answer_start_idx:] = input_ids[answer_start_idx:]
        else:
            # Si no encontramos el marcador, usar todo como label
            label = input_ids.copy()
        labels.append(label)
    
    # Crear dataset
    dataset_dict = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"  Dataset preparado: {len(dataset)} ejemplos", flush=True)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma 3 270M con LoRA')
    parser.add_argument('--dataset-file', type=str, default='docs/lora_dataset.jsonl',
                       help='Archivo JSONL con el dataset')
    parser.add_argument('--base-model', type=str, default='google/gemma-3-270m',
                       help='Ruta al modelo base Gemma o nombre en Hugging Face (default: google/gemma-3-270m)')
    parser.add_argument('--output-dir', type=str, default='models/g3_confirm',
                       help='Directorio donde guardar el modelo fine-tuneado (default: models/g3_confirm)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Longitud máxima de secuencia (default: 512)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Tamaño de batch (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--num-epochs', type=int, default=5,
                       help='Número de épocas (default: 5)')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='Rank de LoRA (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='Alpha de LoRA (default: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='Dropout de LoRA (default: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.05,
                       help='Proporción de datos para test (default: 0.05)')
    parser.add_argument('--save-steps', type=int, default=1000,
                       help='Guardar checkpoint cada N pasos (default: 1000)')
    parser.add_argument('--eval-steps', type=int, default=1000,
                       help='Evaluar cada N pasos (default: 1000)')
    parser.add_argument('--logging-steps', type=int, default=100,
                       help='Logging cada N pasos (default: 100)')
    
    args = parser.parse_args()
    
    print("="*60, flush=True)
    print("FINE-TUNING GEMMA 3 270M", flush=True)
    print("="*60, flush=True)
    
    # Verificar GPU (requerido)
    if not torch.cuda.is_available():
        print("\nERROR: No se detectó GPU disponible.", flush=True)
        print("  El entrenamiento requiere GPU para ejecutarse.", flush=True)
        print("  Verifica que CUDA esté instalado y configurado correctamente.", flush=True)
        return 1
    
    device = "cuda"
    print(f"\nDispositivo: {device}", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
    
    # Verificar memoria disponible
    torch.cuda.empty_cache()
    memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
    print(f"  Memoria GPU libre: {memory_free:.2f} GB", flush=True)
    
    # Cargar dataset
    print(f"\n1. Cargando dataset...", flush=True)
    data = load_dataset(args.dataset_file)
    
    if len(data) == 0:
        print("ERROR: El dataset está vacío", flush=True)
        return 1
    
    # Dividir en train/test
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=args.test_size, random_state=42)
    print(f"  Train: {len(train_data)} ejemplos", flush=True)
    print(f"  Test: {len(test_data)} ejemplos", flush=True)
    
    # Cargar modelo y tokenizador
    print(f"\n2. Cargando modelo base desde {args.base_model}...", flush=True)
    
    # Verificar si es una ruta local o un nombre de modelo en Hugging Face
    is_local_path = os.path.exists(args.base_model)
    
    if is_local_path:
        print(f"  Usando modelo local: {args.base_model}", flush=True)
        model_path = args.base_model
    else:
        print(f"  Descargando modelo desde Hugging Face: {args.base_model}", flush=True)
        print("  Esto puede tardar varios minutos la primera vez...", flush=True)
        model_path = args.base_model
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Siempre usar float16 en GPU
        device_map="auto",  # Distribución automática en GPU
        trust_remote_code=True
    )
    
    print("  Modelo cargado", flush=True)
    
    # Configurar LoRA
    print(f"\n3. Configurando LoRA...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preparar datasets
    print(f"\n4. Preparando datasets...", flush=True)
    train_dataset = prepare_dataset(train_data, tokenizer, max_length=args.max_length)
    test_dataset = prepare_dataset(test_data, tokenizer, max_length=args.max_length)
    
    # Configurar entrenamiento
    print(f"\n5. Configurando entrenamiento...", flush=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,  # Usar precisión mixta en GPU
        dataloader_pin_memory=True,  # Acelerar carga de datos en GPU
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=100,
        report_to="none",  # Desactivar wandb/tensorboard por defecto
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, no masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Entrenar
    print(f"\n6. Iniciando entrenamiento...", flush=True)
    print(f"  Épocas: {args.num_epochs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Learning rate: {args.learning_rate}", flush=True)
    print(f"  Pasos totales: ~{len(train_dataset) // args.batch_size * args.num_epochs}", flush=True)
    print("="*60, flush=True)
    
    trainer.train()
    
    # Guardar modelo final
    print(f"\n7. Guardando modelo final en {args.output_dir}...", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "="*60, flush=True)
    print("ENTRENAMIENTO COMPLETADO", flush=True)
    print("="*60, flush=True)
    print(f"Modelo guardado en: {os.path.abspath(args.output_dir)}", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

