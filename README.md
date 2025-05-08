# CNN-Turing-Complexity

<div align="center">

![Jerarquía de Chomsky](https://miro.medium.com/max/1400/1*gBOXRSYG1SerR8BFGq11xQ.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/)

**Explorando las fronteras teóricas entre Redes Neuronales Convolucionales y la Teoría de la Computabilidad**

</div>

---

## 📚 Acerca del Proyecto

Este repositorio contiene la implementación del código para el artículo **"Fundamentos Matemáticos Aplicados a Sistemas De Redes Neuronales Convolucionales y las Máquinas de Turing"** publicado en la revista EIA. La investigación establece conexiones formales entre la profundidad de las CNNs y su capacidad expresiva en términos de la teoría de la computabilidad.

### 🔍 Motivación

Las redes neuronales profundas han revolucionado el aprendizaje automático, pero sus fundamentos teóricos en relación con modelos computacionales clásicos siguen siendo un área de investigación abierta. Este proyecto aborda la pregunta: **¿Existe una correspondencia entre la profundidad de las redes neuronales y la jerarquía de Chomsky de complejidad computacional?**

### 🔬 Contribución Científica

Este trabajo constituye un puente entre dos campos fundamentales:

1. **Aprendizaje Profundo**: Arquitecturas CNN modernas con bloques residuales
2. **Teoría de la Computación**: Clases de complejidad computacional según la jerarquía de Chomsky

A través de experimentos rigurosos y análisis matemático, demostramos que:

- Las CNNs más profundas pueden reconocer patrones de mayor complejidad computacional
- El tiempo de aprendizaje escala con la complejidad teórica del problema
- Existen límites fundamentales derivados de la teoría de la computabilidad que se manifiestan empíricamente

## 🧠 Fundamento Teórico

<div align="center">
<img src="https://miro.medium.com/max/1400/1*i0o8mjFfCn-uD79-F1Cqkw.png" alt="Relación entre Profundidad y Complejidad" width="600"/>
</div>

La jerarquía de Chomsky clasifica los lenguajes formales en cuatro tipos, cada uno correspondiente a un modelo computacional con diferente poder expresivo:

| Tipo | Clase de Lenguaje | Modelo Computacional | Ejemplo | Implementación |
|------|-------------------|----------------------|---------|----------------|
| 3 | Regular | Autómata Finito | a*b* | Patrones regulares |
| 2 | Libre de Contexto | Autómata con Pila | a^n b^n | Patrones Sierpinski, paréntesis balanceados |
| 1 | Sensible al Contexto | Autómata Limitado Lineal | a^n b^n c^n | Patrones sensibles al contexto |
| 0 | Recursivamente Enumerable | Máquina de Turing | Problema de la parada | Patrones recursivos generales |

Nuestra hipótesis central es que la profundidad arquitectónica de las CNNs se corresponde con esta jerarquía, permitiendo que redes más profundas capturen patrones de mayor complejidad computacional.

## 🚀 Características

- **Generadores de Patrones**: Implementación de generadores para diferentes clases de complejidad
  - Patrones de Sierpinski (recursivos)
  - Patrones de paréntesis balanceados (libres de contexto)
  - Patrones sensibles al contexto (a^n b^n c^n)
  - Patrones regulares y de autómatas linealmente acotados
  
- **Arquitectura CNN Avanzada**:
  - Bloques residuales para mejor flujo de gradiente
  - Profundidad configurable para experimentación sistemática
  - Normalización por lotes y otras técnicas modernas
  
- **Marco Experimental Completo**:
  - Entrenamiento con validación cruzada
  - Análisis de rendimiento por clase de complejidad
  - Visualización detallada de resultados

## 🛠️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/CNN-Turing-Complexity.git
cd CNN-Turing-Complexity

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 📊 Uso

### Ejecutar Experimentos

```bash
# Experimento completo con múltiples profundidades
python scripts/experiment.py --modo completo --num_epocas 30 --profundidades 3 5 7

# Experimento simplificado (rápido)
python scripts/experiment.py --modo simplificado --num_epocas 10
```

### Parámetros Disponibles

| Parámetro | Descripción | Valor Predeterminado |
|-----------|-------------|----------------------|
| `--modo` | Modo de experimento (`simplificado` o `completo`) | `simplificado` |
| `--num_muestras` | Número de muestras en el conjunto de datos | 400 |
| `--tamano_imagen` | Tamaño de las imágenes generadas | 64 |
| `--tamano_lote` | Tamaño del lote para entrenamiento | 32 |
| `--num_epocas` | Número de épocas de entrenamiento | 30 |
| `--tasa_aprendizaje` | Tasa de aprendizaje | 0.001 |
| `--profundidades` | Profundidades de CNN para experimentar | [3, 5, 7] |
| `--semilla` | Semilla aleatoria para reproducibilidad | 42 |
| `--directorio_salida` | Directorio para guardar resultados | `resultados` |
| `--usar_aumento_datos` | Usar aumento de datos para entrenamiento | `False` |
| `--paciencia` | Paciencia para detención temprana | 5 |
| `--num_clases` | Número de clases a utilizar (3-5) | 4 |

### Visualización de Resultados

```bash
# Analizar resultados generados (crea visualizaciones en el directorio de resultados)
python scripts/visualize_results.py --directorio_resultados resultados
```

## 📈 Resultados

<div align="center">
<img src="https://miro.medium.com/max/1400/1*gZ4m2DAz6dlGXKGRrE7o3g.jpeg" alt="Resultados Experimentales" width="700"/>
</div>

Nuestros experimentos muestran una clara correlación entre la profundidad de la CNN y su capacidad para reconocer patrones de diferentes complejidades computacionales:

- **CNNs superficiales (profundidad 3)**: Excelente rendimiento en patrones regulares (Tipo 3), pero rendimiento limitado en patrones más complejos
- **CNNs intermedias (profundidad 5)**: Buena capacidad para reconocer patrones libres de contexto (Tipo 2)
- **CNNs profundas (profundidad 7+)**: Capacidad para reconocer incluso patrones sensibles al contexto (Tipo 1)

Los tiempos de entrenamiento también escalan con la complejidad teórica, validando nuestra hipótesis sobre la relación entre recursos computacionales y complejidad.

## 📝 Publicación Asociada

Si utilizas este código en tu investigación, por favor cita nuestro artículo:

```bibtex
@article{rodriguez2023fundamentos,
  title={Fundamentos Matemáticos Aplicados a Sistemas De Redes Neuronales Convolucionales y las Máquinas de Turing},
  author={Rodríguez Díaz, Nicolás},
  journal={Revista EIA},
  year={2023}
}
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas y apreciadas. Aquí hay algunas formas de contribuir:

- 🐛 Reportar bugs y problemas
- 💡 Proponer nuevas características o mejoras
- 📚 Mejorar la documentación
- 🧪 Agregar más pruebas
- 🔍 Revisar pull requests

Para contribuir:

1. Haz fork del proyecto
2. Crea una rama para tu contribución (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -m 'Agrega nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📧 Contacto

Nicolás Rodríguez Díaz - [nrodrigued@eafit.edu.co](mailto:nrodrigued@eafit.edu.co)

Universidad EAFIT, Medellín, Colombia

---

<div align="center">
<p>
<a href="https://github.com/tu-usuario/CNN-Turing-Complexity/stargazers">⭐ Dame una estrella si te resulta útil! ⭐</a>
</p>

<p>
<b>Una investigación en la intersección del Aprendizaje Profundo y la Teoría de la Computación</b>
</p>
</div>
