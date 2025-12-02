# Experiment Log (2025-12-02)

Entorno: `source .venv/bin/activate` (PyTorch 2.5.0+cu124, TorchVision 0.20.0+cu124, MedMNIST 3.0.2, CUDA disponible).  
Comando base usado en todos los runs (cambiando `--dataset-name`):  
`python main.py --dataset-name <dataset> --max-generations 1 --num-epochs 1 --batch-size 32 --learning-rate 0.001 --reef-size "(2,2)" --mutation-rate 0.1 --fitness-method linear --fitness-alpha 7 --fitness-beta 1.0 --patience 2 --no-shuffle-dataset`

## Resultados por dataset

- mnist — **OK** — duración 62s — accuracy test 91.42%, fitness 0.9018, params ~124K.  
  - Exp dir: `experiments/mnist_InceptionCRO_20251202_211833`.

- chestmnist — **Fallo** — duración 17s — error: `RuntimeError: Expected floating point type for target with class probabilities, got Long` al usar `CrossEntropyLoss` con targets one-hot/multi-label.  
  - Hipótesis: falta diferenciar tareas multi-label y usar BCEWithLogits + transformación de etiquetas.

- pathmnist — **Fallo** — duración 22s — error: `RuntimeError: 0D or 1D target tensor expected, multi-target not supported` (targets multi-dim con CrossEntropy).  
  - Hipótesis: dataset entrega vectores/one-hot; se requiere conversión a índices o pérdida adecuada.

- dermamnist — **Fallo** — duración 14s — mismo error de tensor multi-dim para CrossEntropy.

- octmnist — **Fallo** — duración 29s — mismo error de tensor multi-dim para CrossEntropy.

- pneumoniamnist — **Fallo** — duración 13s — mismo error de tensor multi-dim para CrossEntropy.

- retinamnist — **Fallo** — duración 14s — mismo error de tensor multi-dim para CrossEntropy.

- breastmnist — **Fallo** — duración 14s — mismo error de tensor multi-dim para CrossEntropy.

- bloodmnist — **Fallo** — duración 19s — mismo error de tensor multi-dim para CrossEntropy.

- tissuemnist — **Fallo** — duración 29s — mismo error de tensor multi-dim para CrossEntropy (tras descarga completa ~125MB).

- organamnist — **Fallo** — duración 17s — mismo error de tensor multi-dim para CrossEntropy.

- organcmnist — **Fallo** — duración 16s — mismo error de tensor multi-dim para CrossEntropy.

- organsmnist — **Fallo** — duración 19s — mismo error de tensor multi-dim para CrossEntropy.

## Observaciones generales

- Únicamente MNIST funciona end-to-end con la configuración actual.
- Todos los subsets de MedMNIST fallan en la fase de entrenamiento completo debido a incompatibilidad entre las etiquetas devueltas (multi-dim/one-hot o multi-label) y el uso fijo de `CrossEntropyLoss` en `trainer.py`. El mensaje cambia entre “targets con probabilidades” y “multi-target no soportado”.
- Se generaron directorios bajo `experiments/<dataset>_InceptionCRO_*` y plots CSV correspondientes; están ignorados por git.

## Próximos pasos recomendados

1. Introducir manejo de tipo de tarea (multi-clase vs multi-label) en `load_medmnist`/`trainer.py`, eligiendo la pérdida (CrossEntropy vs BCEWithLogits) y normalizando etiquetas (argmax a índices o float multi-label).
2. Repetir la batería mínima tras el ajuste para confirmar soporte de todos los subsets MedMNIST.
