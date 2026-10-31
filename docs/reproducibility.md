# Reproducibilidad

## Entorno de desarrollo

- Sistema: Windows 11
- Entorno: Conda `dm1`
- Python: 3.13.2
- CPU: Intel Core i9-14900K
- GPU: NVIDIA GeForce RTX 3050, 8 GB
- Memoria RAM: 96 GB
- Controlador NVIDIA: 610.88
- PyTorch: 2.14.0, wheel CUDA 12.6

PyTorch detecta la GPU correctamente con `torch.cuda.is_available() == True`.

## Reproducibilidad experimental

- Semilla inicial: `42`
- Partición de validación por motor, con semilla fija.
- El escalador se ajustará únicamente con motores de entrenamiento.
- La misma partición y configuración se utilizarán para comparar MLP y LSTM.
- Las semillas no garantizan resultados bit a bit idénticos entre versiones de software y hardware.
