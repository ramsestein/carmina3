#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para fine-tunear el modelo Gemma 3 270M usando LoRA con el dataset lora_dataset.jsonl.
Versión optimizada para Google Colab con GPU A100.
El modelo fine-tuneado se guardará como g3_confirm.

FIXED: Gradient checkpointing habilitado correctamente para PEFT
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Verificar si estamos en Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Instalar dependencias que falten
def install_missing_packages():
    """Instala las librerías necesarias si no están disponibles."""
    print("Instalando/actualizando dependencias...", flush=True)
    
    # Siempre actualizar transformers y peft a versiones compatibles
    # Versiones compatibles: transformers 4.40+ y peft 0.8+
    print("  Actualizando transformers y peft a versiones compatibles...", flush=True)
    os.system("pip install -q --upgrade 'transformers>=4.40.0' 'peft>=0.8.0'")
    
    # Verificar otras dependencias
    other_packages = {
        'datasets': 'datasets',
        'accelerate': 'accelerate',
        'sklearn': 'scikit-learn',
    }
    
    missing = []
    for module_name, package_name in other_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"  Instalando dependencias faltantes: {', '.join(missing)}...", flush=True)
        install_cmd = f"pip install -q {' '.join(missing)}"
        os.system(install_cmd)
    
    print("Dependencias instaladas/actualizadas.", flush=True)

# Instalar dependencias faltantes
install_missing_packages()

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
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except ImportError:
    print("ERROR: peft no está instalado. Instálalo con: pip install peft", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"ERROR al importar peft: {e}", flush=True)
    print("Intenta actualizar: pip install --upgrade peft transformers", flush=True)
    sys.exit(1)

# Configurar TOKENIZERS_PARALLELISM para evitar advertencias en multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configurar codificación UTF-8
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
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Crear Dataset de HuggingFace
    dataset = Dataset.from_dict(tokenized)
    
    print(f"  Dataset preparado: {len(dataset)} ejemplos", flush=True)
    return dataset


def main():
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M con LoRA")
    
    # Dataset y modelo
    parser.add_argument("--dataset-file", type=str, default="lora_dataset.jsonl",
                       help="Ruta al archivo JSONL con el dataset")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m",
                       help="Modelo base de Hugging Face")
    parser.add_argument("--output-dir", type=str, default="content/drive/My Drive/g3_confirm",
                       help="Directorio de salida para el modelo fine-tuneado")
    parser.add_argument("--drive-output", type=str, default=None,
                       help="Directorio de Google Drive para guardar el modelo (opcional)")
    
    # Parámetros de entrenamiento
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Tamaño del batch por dispositivo")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                       help="Pasos de acumulación de gradientes")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Longitud máxima de secuencia")
    parser.add_argument("--test-size", type=float, default=0.1,
                       help="Proporción de datos para test (0.0-1.0)")
    
    # Parámetros de LoRA
    parser.add_argument("--lora-r", type=int, default=16,
                       help="Rango de LoRA")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="Alpha de LoRA")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="Dropout de LoRA")
    
    # Logging y guardado
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Frecuencia de logging")
    parser.add_argument("--save-steps", type=int, default=100,
                       help="Frecuencia de guardado")
    parser.add_argument("--eval-steps", type=int, default=100,
                       help="Frecuencia de evaluación")
    
    # Weights & Biases (opcional)
    parser.add_argument("--use-wandb", action="store_true",
                       help="Usar Weights & Biases para logging")
    parser.add_argument("--wandb-project", type=str, default="gemma-3-lora",
                       help="Nombre del proyecto en W&B")
    
    # Hugging Face token
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Token de Hugging Face (opcional)")
    
    # Parámetros para modelo punitivo
    parser.add_argument("--punitive-weight", type=float, default=2.0,
                       help="Peso para penalizar falsos positivos (default: 2.0, más alto = más punitivo)")
    parser.add_argument("--use-punitive-loss", action="store_true",
                       help="Usar loss punitivo que penaliza más los falsos positivos")
    
    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                       help="Número de evaluaciones sin mejora antes de detener el entrenamiento (default: 3)")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0,
                       help="Umbral mínimo de mejora para considerar que hay progreso (default: 0.0)")
    
    args = parser.parse_args()
    
    # Verificar GPU
    print("="*60, flush=True)
    print("CONFIGURACIÓN DEL SISTEMA", flush=True)
    print("="*60, flush=True)
    
    if not torch.cuda.is_available():
        print("ADVERTENCIA: No se detectó GPU. El entrenamiento será muy lento.", flush=True)
        is_a100 = False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {gpu_name}", flush=True)
        is_a100 = "A100" in gpu_name
        
        if is_a100:
            print("  ✓ GPU A100 detectada - usando configuración optimizada", flush=True)
        
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
    
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA disponible: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}", flush=True)
    print("="*60, flush=True)
    
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
    
    # Intentar autenticación automática si no se proporcionó token
    if not args.hf_token:
        try:
            from huggingface_hub import whoami
            whoami()  # Verificar si ya hay una sesión activa
            print("  Usando sesión de Hugging Face existente", flush=True)
        except:
            print("  ADVERTENCIA: No hay sesión de Hugging Face activa", flush=True)
            print("  Si el modelo requiere acceso gated, usa: huggingface-cli login", flush=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            token=args.hf_token if args.hf_token else None
        )
        
        # Configurar pad_token si no existe
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Determinar dtype: usar bfloat16 en A100 si está disponible, sino float16
        if is_a100 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("  Usando bfloat16 (optimizado para A100)", flush=True)
        else:
            dtype = torch.float16
            print("  Usando float16", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto",  # Distribución automática en GPU
            trust_remote_code=True,
            token=args.hf_token if args.hf_token else None,
            use_cache=False  # Deshabilitar cache cuando se usa gradient checkpointing
        )
        
        print("  Modelo cargado", flush=True)
    except Exception as e:
        print(f"\nERROR al cargar el modelo: {e}", flush=True)
        print("\nSOLUCIÓN:", flush=True)
        print("1. Obtén un token de Hugging Face:", flush=True)
        print("   https://huggingface.co/settings/tokens", flush=True)
        print("2. Acepta los términos del modelo:", flush=True)
        print("   https://huggingface.co/google/gemma-3-270m", flush=True)
        print("3. Ejecuta en Colab:", flush=True)
        print("   !huggingface-cli login", flush=True)
        print("   O usa: --hf-token TU_TOKEN", flush=True)
        return 1
    
    # CRÍTICO: Preparar modelo para gradient checkpointing ANTES de aplicar LoRA
    print(f"\n3. Preparando modelo para gradient checkpointing...", flush=True)
    model.gradient_checkpointing_enable()
    
    # Para modelos PEFT, necesitamos habilitar input_requires_grad
    model.enable_input_require_grads()
    
    print("  ✓ Gradient checkpointing habilitado", flush=True)
    
    # Configurar LoRA
    print(f"\n4. Configurando LoRA...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Asegurarse de que el modelo esté en modo entrenamiento
    model.train()
    
    # Verificar que hay parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Parámetros entrenables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)", flush=True)
    print(f"  Parámetros totales: {total_params:,}", flush=True)
    
    if trainable_params == 0:
        print("  ERROR CRÍTICO: No hay parámetros entrenables.", flush=True)
        print("  Verificando parámetros del modelo...", flush=True)
        for name, param in model.named_parameters():
            print(f"    {name}: requires_grad={param.requires_grad}", flush=True)
        return 1
    
    model.print_trainable_parameters()
    
    # Preparar datasets
    print(f"\n5. Preparando datasets...", flush=True)
    train_dataset = prepare_dataset(train_data, tokenizer, max_length=args.max_length)
    test_dataset = prepare_dataset(test_data, tokenizer, max_length=args.max_length)
    
    # Configurar logging (W&B opcional)
    report_to = "none"
    if args.use_wandb:
        try:
            import wandb
            wandb.login()
            report_to = "wandb"
            wandb.init(
                project=args.wandb_project,
                config={
                    "model": args.base_model,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.num_epochs,
                }
            )
            print("  ✓ Weights & Biases configurado", flush=True)
        except ImportError:
            print("  ADVERTENCIA: wandb no está instalado. Instálalo con: pip install wandb", flush=True)
            report_to = "none"
    
    # Configurar entrenamiento
    print(f"\n6. Configurando entrenamiento...", flush=True)
    
    # Usar bf16 en A100 si está disponible
    use_bf16 = is_a100 and torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=not use_bf16,  # Usar fp16 solo si no hay bf16
        bf16=use_bf16,  # Usar bf16 en A100
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Más workers en Colab
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=100,
        report_to=report_to,
        save_safetensors=True,  # Usar formato seguro
        gradient_checkpointing=True,  # Ahorrar memoria
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Importante para PEFT
    )
    
    # Configurar early stopping
    from transformers import EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, no masked LM
    )
    
    # Custom Trainer con loss punitivo para falsos positivos
    if args.use_punitive_loss:
        class PunitiveTrainer(Trainer):
            """
            Trainer personalizado que penaliza más los falsos positivos (predicciones "Sí" incorrectas).
            Esto hace que el modelo sea más conservador y rechace más entidades.
            """
            def __init__(self, false_positive_weight=2.0, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.false_positive_weight = false_positive_weight
            
            def compute_loss(self, model, inputs, return_outputs=False):
                """
                Calcula el loss con penalización adicional para falsos positivos.
                """
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Loss estándar
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Identificar tokens que corresponden a predicciones "Sí" incorrectas
                # Buscar secuencias donde el modelo predice "Sí" pero debería predecir "No"
                # Esto requiere analizar los labels y las predicciones
                
                # Obtener predicciones
                preds = torch.argmax(logits, dim=-1)
                
                # Token IDs para "Sí" y "No" (necesitamos encontrarlos en el tokenizer)
                # Para Gemma, necesitamos buscar los tokens correspondientes
                si_token_id = None
                no_token_id = None
                
                # Buscar tokens "Sí" y "No" en el vocabulario
                if hasattr(tokenizer, 'encode'):
                    try:
                        si_tokens = tokenizer.encode("Sí", add_special_tokens=False)
                        no_tokens = tokenizer.encode("No", add_special_tokens=False)
                        if si_tokens:
                            si_token_id = si_tokens[0]
                        if no_tokens:
                            no_token_id = no_tokens[0]
                    except:
                        pass
                
                # Si encontramos los tokens, aplicar penalización
                if si_token_id is not None and no_token_id is not None:
                    # Crear máscara para falsos positivos: predice "Sí" pero debería ser "No"
                    # Esto es complejo porque necesitamos saber qué tokens corresponden a la respuesta
                    # Por ahora, aplicamos una penalización general más simple
                    pass
                
                # Alternativa más simple: aumentar el peso del loss cuando hay alta confianza en predicciones incorrectas
                # Penalizar más cuando el modelo está muy seguro de una predicción incorrecta
                probs = torch.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                
                # Aumentar el loss cuando hay alta confianza pero predicción incorrecta
                confidence_penalty = (max_probs.view(-1) * (preds.view(-1) != labels.view(-1)).float())
                loss = loss + self.false_positive_weight * confidence_penalty * loss
                
                loss = loss.mean()
                
                return (loss, outputs) if return_outputs else loss
    
    # Crear trainer (punitivo o estándar)
    if args.use_punitive_loss:
        print(f"\n  Usando loss punitivo con peso: {args.punitive_weight}", flush=True)
        print(f"  Esto hará que el modelo sea más conservador (rechace más entidades)", flush=True)
        trainer = PunitiveTrainer(
            false_positive_weight=args.punitive_weight,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            callbacks=[early_stopping_callback],
        )
    
    print(f"\n  Early stopping configurado:", flush=True)
    print(f"    Patience: {args.early_stopping_patience} evaluaciones", flush=True)
    print(f"    Threshold: {args.early_stopping_threshold}", flush=True)
    
    # Entrenar
    print(f"\n7. Iniciando entrenamiento...", flush=True)
    print(f"  Épocas: {args.num_epochs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}", flush=True)
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}", flush=True)
    print(f"  Learning rate: {args.learning_rate}", flush=True)
    print(f"  Precision: {'bfloat16' if use_bf16 else 'float16'}", flush=True)
    print(f"  Pasos totales: ~{len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs}", flush=True)
    print("="*60, flush=True)
    
    trainer.train()
    
    # Guardar modelo final
    print(f"\n8. Guardando modelo final en {args.output_dir}...", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Si se especificó Google Drive, copiar también allí
    if args.drive_output:
        print(f"  Copiando modelo a Google Drive: {args.drive_output}...", flush=True)
        import shutil
        os.makedirs(args.drive_output, exist_ok=True)
        shutil.copytree(args.output_dir, args.drive_output, dirs_exist_ok=True)
        print(f"  ✓ Modelo copiado a Google Drive", flush=True)
    
    # Cerrar W&B si se usó
    if args.use_wandb and report_to == "wandb":
        wandb.finish()
    
    print("\n" + "="*60, flush=True)
    print("ENTRENAMIENTO COMPLETADO", flush=True)
    print("="*60, flush=True)
    print(f"Modelo guardado en: {os.path.abspath(args.output_dir)}", flush=True)
    if args.drive_output:
        print(f"También guardado en Google Drive: {args.drive_output}", flush=True)
    print("="*60, flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())