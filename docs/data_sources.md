# Fuente y procedencia de los datos

## NASA C-MAPSS FD001

- Catálogo del conjunto de datos: https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data
- Descarga oficial usada por el pipeline: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Repositorio de datos de NASA PCoE: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

El pipeline conserva `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` y `readme.txt` cuando está disponible. No altera los archivos descargados. `data/raw/data_manifest.json` registra la URL, fecha de descarga, tamaño y SHA-256 del ZIP, además del tamaño y SHA-256 de cada archivo extraído.

El catálogo y el `readme.txt` describen datos simulados, 100 trayectorias para train y 100 para test en FD001, una condición operativa y un modo de degradación HPC. El test está truncado y `RUL_FD001.txt` contiene el RUL al final de las trayectorias observadas, en orden de `unit_id` del 1 al 100.

La página de NASA PCoE solicita reconocer el repositorio y a quienes donaron los datos; también indica que los usuarios emplean los datos bajo su propio riesgo y que NASA y los donantes no asumen responsabilidad por su uso. No se encontró una licencia explícita en la página del repositorio ni en el `readme.txt` incluido. No se atribuye una licencia al conjunto.

### Citas

Saxena, A., y Goebel, K. (2008). “Turbofan Engine Degradation Simulation Data Set”. NASA Prognostics Data Repository, NASA Ames Research Center.

Saxena, A., Goebel, K., Simon, D., y Eklund, N. (2008). “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”. *Proceedings of the 1st International Conference on Prognostics and Health Management*.
