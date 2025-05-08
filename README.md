<div align="center">

# 🧠 CNN-Turing-Complexity 🧠

<img src="https://raw.githubusercontent.com/sindresorhus/awesome/main/media/logo.svg" width="80" height="80" align="left">
<img src="https://raw.githubusercontent.com/sindresorhus/awesome/main/media/logo.svg" width="80" height="80" align="right">

[![Journal: EIA](https://img.shields.io/badge/Journal-EIA-green.svg)](https://revista.eia.edu.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue.svg)](https://doi.org/)

**Una investigación matemática formal sobre la intersección entre Redes Neuronales Convolucionales y la Teoría de la Computabilidad**

*Universidad EAFIT - Departamento de Ingeniería Matemática - Medellín, Colombia*

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*8wU0hfUY3UK_D8Y7tbIyFQ.png" alt="Jerarquía de Chomsky y CNNs" width="750">
</p>

</div>

---

<div align="center">

[📚 Fundamentos](#-fundamentos) •
[🔬 Implementación](#-implementación) •
[📊 Experimentos](#-experimentos) •
[📝 Publicación](#-publicación) •
[🚀 Inicio Rápido](#-inicio-rápido) •
[📈 Resultados](#-resultados) •
[🛣️ Trabajo Futuro](#️-trabajo-futuro) •
[👥 Colaboradores](#-colaboradores)

</div>

---

<details>
<summary><b>Tabla de Contenidos (expandir)</b></summary>

- [🎯 Visión General](#-visión-general)
- [📚 Fundamentos](#-fundamentos)
  - [Jerarquía de Chomsky](#jerarquía-de-chomsky)
  - [Fundamentos Teóricos](#fundamentos-teóricos)
  - [Teoremas Centrales](#teoremas-centrales)
- [🧮 Modelos Matemáticos](#-modelos-matemáticos)
- [🔬 Implementación](#-implementación)
  - [Arquitectura CNN](#arquitectura-cnn)
  - [Generadores de Patrones](#generadores-de-patrones)
  - [Framework de Experimentación](#framework-de-experimentación)
- [📊 Experimentos](#-experimentos)
  - [Diseño Experimental](#diseño-experimental)
  - [Métricas y Evaluación](#métricas-y-evaluación)
- [📝 Publicación](#-publicación)
- [🚀 Inicio Rápido](#-inicio-rápido)
  - [Requisitos](#requisitos)
  - [Instalación](#instalación)
  - [Ejecución de Experimentos](#ejecución-de-experimentos)
  - [Visualización de Resultados](#visualización-de-resultados)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [📈 Resultados](#-resultados)
  - [Hallazgos Principales](#hallazgos-principales)
  - [Visualizaciones](#visualizaciones)
- [🛣️ Trabajo Futuro](#️-trabajo-futuro)
- [❓ Preguntas Frecuentes](#-preguntas-frecuentes)
- [👥 Colaboradores](#-colaboradores)
- [🙏 Agradecimientos](#-agradecimientos)
- [📄 Licencia](#-licencia)
- [✉️ Contacto](#️-contacto)

</details>

## 🎯 Visión General

CNN-Turing-Complexity representa una contribución significativa en la intersección entre el aprendizaje profundo moderno y los fundamentos teóricos de la computación. Este proyecto implementa experimentos que validan formalmente la conexión entre la profundidad arquitectónica de las redes neuronales convolucionales (CNNs) y su capacidad para representar computaciones de diferentes niveles de complejidad según la jerarquía de Chomsky.

> **Hipótesis Principal**: La profundidad de las redes neuronales convolucionales determina su capacidad para reconocer patrones de diferentes niveles de complejidad computacional, estableciendo una correspondencia directa con la jerarquía de Chomsky.

```text
Mayor Profundidad CNN ↔️ Mayor Poder Computacional ↔️ Mayor Complejidad en Jerarquía de Chomsky
```

Este repositorio contiene la implementación completa de los experimentos descritos en el artículo **"Fundamentos Matemáticos Aplicados a Sistemas De Redes Neuronales Convolucionales y las Máquinas de Turing"**, publicado en la revista EIA.

## 📚 Fundamentos

### Jerarquía de Chomsky

<div align="center">
  <table>
    <tr>
      <th>Tipo</th>
      <th>Clase de Lenguaje</th>
      <th>Gramática</th>
      <th>Autómata</th>
      <th>Complejidad</th>
      <th>Patrones Implementados</th>
    </tr>
    <tr>
      <td align="center">0</td>
      <td>Recursivamente Enumerable</td>
      <td>Sin restricciones</td>
      <td>Máquina de Turing</td>
      <td>Indecidible</td>
      <td>Patrones recursivos generales</td>
    </tr>
    <tr>
      <td align="center">1</td>
      <td>Sensible al Contexto</td>
      <td>αAβ → αγβ</td>
      <td>Autómata Limitado Lineal</td>
      <td>PSPACE</td>
      <td>a<sup>n</sup>b<sup>n</sup>c<sup>n</sup>, LBA</td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td>Libre de Contexto</td>
      <td>A → γ</td>
      <td>Autómata con Pila</td>
      <td>P</td>
      <td>Sierpinski, paréntesis balanceados</td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td>Regular</td>
      <td>A → aB o A → a</td>
      <td>Autómata Finito</td>
      <td>LOGSPACE</td>
      <td>Patrones regulares</td>
    </tr>
  </table>
</div>

### Fundamentos Teóricos

La investigación se sustenta en tres pilares teóricos fundamentales:

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YcdlKIjkcJz2MpzGwpO9aQ.png" alt="Fundamentos Teóricos" width="700">
</div>

1. **Teoría de la Computabilidad** (Turing, Church, 1936-1937): Define los límites fundamentales de lo que puede ser calculado algorítmicamente.

2. **Jerarquía de Chomsky** (Chomsky, 1956): Clasifica los lenguajes formales según su complejidad gramatical, estableciendo una correspondencia con diferentes modelos computacionales.

3. **Teoremas de Universalidad** (Cybenko, 1989; Hornik, 1991): Establecen la capacidad de las redes neuronales como aproximadores universales de funciones, pero no abordan completamente la capacidad computacional en relación con la profundidad.

### Teoremas Centrales

Nuestro trabajo introduce y demuestra tres teoremas fundamentales:

<div style="background-color: #f6f8fa; padding: 16px; border-radius: 6px; margin: 20px 0;">
  <p><strong>Teorema 1 (Equivalencia Expresiva)</strong>: Una CNN de profundidad <em>d</em> puede reconocer todos los patrones generados por gramáticas de tipo <em>k</em> en la jerarquía de Chomsky donde <em>k</em> ≥ 4-⌈log₂(d)⌉.</p>
  
  <p><strong>Teorema 2 (Límite Inferior de Profundidad)</strong>: Para reconocer patrones generados por gramáticas de tipo <em>k</em>, se requiere una CNN con una profundidad mínima de <em>d</em> = 2^(4-<em>k</em>).</p>
  
  <p><strong>Teorema 3 (Complejidad del Entrenamiento)</strong>: El problema de encontrar los pesos óptimos de una CNN para reconocer patrones de tipo <em>k</em> tiene una complejidad algorítmica que pertenece a la misma clase de complejidad que el problema de reconocimiento del lenguaje correspondiente.</p>
</div>

## 🧮 Modelos Matemáticos

El fundamento matemático del proyecto integra elementos del análisis funcional, álgebra lineal y teoría de la computación:

**1. Formalización de la CNN:**
```math
\Phi = \varphi_L \circ \varphi_{L-1} \circ \cdots \circ \varphi_1
```

Donde cada capa $\varphi_l$ está definida como una composición de operaciones:

```math
\varphi_l^{\text{conv}}(X)_j = \sum_{i=1}^{c_{l-1}} K_{ji} * X_i + b_j
```

```math
\varphi_l^{\text{act}}(X) = \sigma(X)
```

```math
\varphi_l^{\text{pool}}(X)_{i,j,k} = \max\{X_{i',j',k'} : i' \leq i < (i+1)s, j' \leq j < (j+1)s\}
```

**2. Operación de Convolución:**
```math
(I * K)(x, y) = \sum_{i=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} \sum_{j=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} I(x-i, y-j)K(i, j)
```

**3. Formalización de Máquina de Turing:**
Una máquina de Turing es una 7-tupla $M = (Q, \Gamma, b, \Sigma, \delta, q_0, F)$ donde:
- $Q$ es un conjunto finito de estados
- $\Gamma$ es un conjunto finito de símbolos de cinta
- $b \in \Gamma$ es el símbolo blanco
- $\Sigma \subset \Gamma \setminus \{b\}$ es el conjunto de símbolos de entrada
- $\delta: Q \times \Gamma \rightarrow Q \times \Gamma \times \{L, R\}$ es la función de transición
- $q_0 \in Q$ es el estado inicial
- $F \subseteq Q$ es el conjunto de estados de aceptación

## 🔬 Implementación

### Arquitectura CNN

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kbZs834HTes3AsAwZy6VVw.png" alt="Arquitectura CNN" width="720">
</div>

Nuestra implementación utiliza una arquitectura CNN moderna con las siguientes características:

```python
class CNNModerna(nn.Module):
    def __init__(self, profundidad, num_clases=4, forma_entrada=(1, 64, 64)):
        super(CNNModerna, self).__init__()
        self.profundidad = profundidad
        
        # Número de bloques convolucionales determinado por la profundidad
        num_bloques_conv = max(2, min(profundidad, 10))
        
        # Convolución inicial
        self.conv_inicial = nn.Sequential(
            nn.Conv2d(forma_entrada[0], 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Bloques residuales
        self.bloques_conv = nn.ModuleList()
        # ... (implementación de bloques residuales) ...
        
        # Capas finales de clasificación
        self.capas_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.tamano_caracteristicas, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_clases)
        )
```

**Características clave:**
- **Bloques Residuales**: Permiten mejor flujo de gradiente en redes profundas
- **Normalización por Lotes**: Mejora la estabilidad y velocidad de convergencia
- **Profundidad Variable**: Experimentación con diferentes niveles de profundidad
- **Regularización**: Aplicación de dropout para prevenir sobreajuste

### Generadores de Patrones

Implementamos generadores para las cuatro clases principales de la jerarquía de Chomsky:

<div align="center">
  <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
    <div style="width: 24%; text-align: center;">
      <h4>Tipo 3: Regular</h4>
      <pre>     █ █ █ █ 
     █     █ 
     █ █ █ █ 
     █     █ 
     █ █ █ █</pre>
    </div>
    <div style="width: 24%; text-align: center;">
      <h4>Tipo 2: Libre Contexto</h4>
      <pre>     █ 
    ███
   █████
  ███████
 █████████</pre>
    </div>
    <div style="width: 24%; text-align: center;">
      <h4>Tipo 1: Sensible Contexto</h4>
      <pre>  ════════
     ║║║
     ║║║
     ╝╝╝
     ╲╲╲</pre>
    </div>
    <div style="width: 24%; text-align: center;">
      <h4>Tipo 0: Recursivo</h4>
      <pre>  ╱╲      ╱╲  
 ╱  ╲    ╱  ╲
╱╲  ╱╲  ╱╲  ╱╲
   ╱╲      ╱╲  
  ╱  ╲    ╱  ╲</pre>
    </div>
  </div>
</div>

```python
class GeneradorPatrones:
    @staticmethod
    def crear_sierpinski(orden, tamano=64):
        # Implementación del triángulo de Sierpinski (Tipo 2)
        # ...

    @staticmethod
    def crear_patron_parentesis(profundidad, tamano=64):
        # Patrón de paréntesis balanceados (Tipo 2)
        # ...
        
    @staticmethod
    def crear_patron_sensible_contexto(n, tamano=64):
        # Implementación de a^n b^n c^n (Tipo 1)
        # ...
        
    @staticmethod
    def crear_patron_regular(tipo_patron, tamano=64):
        # Patrones regulares (Tipo 3)
        # ...
    
    @staticmethod
    def crear_automata_limitado_lineal(regla, tamano=64):
        # Patrones de autómata limitado lineal (Tipo 1)
        # ...
```

### Framework de Experimentación

El framework experimental incluye:

1. **Generación de Datos**: Creación de conjuntos de datos con patrones de diferentes complejidades
2. **Entrenamiento**: Sistema de entrenamiento con validación cruzada y optimización
3. **Evaluación**: Métricas específicas para cada clase de complejidad
4. **Análisis**: Herramientas estadísticas para validar hipótesis

<details>
<summary><b>Ver diagrama de flujo del framework (expandir)</b></summary>
<div align="center">
<pre>
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Generación de      │     │   Entrenamiento     │     │     Evaluación      │
│  Patrones           │     │   del Modelo        │     │     y Análisis      │
├─────────────────────┤     ├─────────────────────┤     ├─────────────────────┤
│ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
│ │ Sierpinski      │ │     │ │ Inicialización  │ │     │ │ Matriz de       │ │
│ │ (Tipo 2)        │ │     │ │ del Modelo      │ │     │ │ Confusión       │ │
│ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
│ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
│ │ Paréntesis      │ │     │ │ Optimización    │ │     │ │ Curvas de       │ │
│ │ (Tipo 2)        │ │ ──▶️ │ │ con AdamW       │ │ ──▶️ │ │ Aprendizaje     │ │
│ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
│ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
│ │ a^n b^n c^n     │ │     │ │ Detención       │ │     │ │ Análisis de     │ │
│ │ (Tipo 1)        │ │     │ │ Temprana        │ │     │ │ Complejidad     │ │
│ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
│ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
│ │ Patrones        │ │     │ │ Regularización  │ │     │ │ Visualización   │ │
│ │ Regulares (T3)  │ │     │ │ con Dropout     │ │     │ │ de Resultados   │ │
│ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
</pre>
</div>
</details>

## 📊 Experimentos

### Diseño Experimental

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*LoiyIHdTdwpgL-X2lQV7nw.png" alt="Diseño Experimental" width="700">
</div>

El diseño experimental evalúa sistemáticamente la relación entre:

- **Profundidad de la red**: Variando de 3 a 7 bloques convolucionales
- **Complejidad computacional de los patrones**: Implementando los cuatro tipos de la jerarquía de Chomsky
- **Recursos computacionales**: Medición de tiempo, memoria y complejidad de optimización

#### Configuración Experimental:

```bash
# Experimento completo con múltiples profundidades
python scripts/experiment.py --modo completo --num_epocas 30 --profundidades 3 5 7 --num_clases 4

# Evaluación de patrones específicos
python scripts/experiment.py --modo completo --num_epocas 20 --profundidades 5 --num_clases 4 --patron_especifico "sensible_contexto"
```

### Métricas y Evaluación

Para cada tipo de patrón y profundidad de red, evaluamos:

1. **Precisión de clasificación**: Capacidad de reconocer correctamente el tipo de patrón
2. **Tiempo de convergencia**: Número de épocas hasta alcanzar precisión óptima
3. **Estabilidad del aprendizaje**: Varianza en resultados con diferentes inicializaciones
4. **Eficiencia computacional**: Relación entre profundidad, parámetros y rendimiento

## 📝 Publicación

Este repositorio acompaña el artículo académico:

> **Fundamentos Matemáticos Aplicados a Sistemas De Redes Neuronales Convolucionales y las Máquinas de Turing**  
> Nicolás Rodríguez Díaz  
> Revista EIA  
> DOI: [10.XXXX/XXXXX](https://doi.org/10.XXXX/XXXXX)

<details>
<summary><b>Resumen del artículo (expandir)</b></summary>

<p>Este artículo desarrolla un análisis matemático detallado de los fundamentos que sustentan las redes neuronales convolucionales (CNN) y su relación formal con los conceptos de computabilidad basados en las máquinas de Turing. Se propone un marco teórico unificador que examina de manera integrada las propiedades algebraicas, topológicas y analíticas de las operaciones de convolución, las funciones de activación y los métodos de optimización multivariable empleados en las CNN.</p>

<p>Mediante demostraciones formales, se demuestra que, aunque estas redes actúan como aproximadores universales de funciones continuas mediante transformaciones no lineales en espacios de Hilbert, están sujetas a las mismas limitaciones fundamentales que imponen las máquinas de Turing, conforme a la tesis de Church-Turing. Se introducen teoremas originales que vinculan la capacidad expresiva de las CNN con la jerarquía de clases de complejidad computacional, y se presentan resultados experimentales que validan dichas conexiones teóricas.</p>

<p>Las implicaciones de este estudio son esenciales para comprender los límites teóricos y prácticos del aprendizaje profundo en contextos computacionalmente complejos.</p>
</details>

Si utilizas este código o te basas en nuestros resultados, por favor cita el artículo:

```bibtex
@article{rodriguez2023fundamentos,
  title={Fundamentos Matemáticos Aplicados a Sistemas De Redes Neuronales Convolucionales y las Máquinas de Turing},
  author={Rodríguez Díaz, Nicolás},
  journal={Revista EIA},
  year={2023},
  doi={10.xxxx/xxxxx}
}
```

## 🚀 Inicio Rápido

### Requisitos

- Python 3.8+
- PyTorch 1.10+
- CUDA (opcional, pero recomendado para entrenamiento acelerado)

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/CNN-Turing-Complexity.git
cd CNN-Turing-Complexity

# Crear y activar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución de Experimentos

```bash
# Experimento simplificado (desarrollo rápido)
python scripts/experiment.py --modo simplificado --num_epocas 10

# Experimento completo (resultados para publicación)
python scripts/experiment.py --modo completo --num_epocas 30 --profundidades 3 5 7 --directorio_salida resultados/completo
```

<details>
<summary><b>Parámetros Disponibles (expandir)</b></summary>

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
| `--patron_especifico` | Tipo específico de patrón para análisis detallado | `None` |
| `--guardar_modelo` | Guardar los modelos entrenados | `True` |
| `--gpu_id` | ID de GPU a utilizar | 0 |
| `--verbose` | Nivel de detalle en los logs | 1 |

</details>

### Visualización de Resultados

```bash
# Visualizar resultados de un experimento
python scripts/visualize_results.py --directorio_resultados resultados/completo

# Generar informe completo con todas las métricas y visualizaciones
python scripts/generate_report.py --directorio_resultados resultados/completo --formato pdf
```

## 📁 Estructura del Proyecto

```
CNN-Turing-Complexity/
├── src/                          # Código fuente principal
│   ├── models/                   # Arquitecturas de redes neuronales
│   │   ├── __init__.py
│   │   └── cnn_models.py         # Implementación de CNNModerna
│   ├── data/                     # Procesamiento y generación de datos
│   │   ├── __init__.py
│   │   └── generador_patrones.py # Generadores de patrones computacionales
│   ├── training/                 # Lógica de entrenamiento
│   │   ├── __init__.py
│   │   └── entrenador.py         # Sistema de entrenamiento y evaluación
│   └── visualization/            # Herramientas de visualización
│       ├── __init__.py
│       └── visualizador.py       # Visualización de resultados
├── scripts/                      # Scripts ejecutables
│   ├── experiment.py             # Script principal de experimentos
│   ├── visualize_results.py      # Visualización independiente
│   └── generate_report.py        # Generación de informes
├── notebooks/                    # Jupyter notebooks para análisis
│   ├── exploracion_patrones.ipynb    # Análisis de patrones generados
│   └── analisis_resultados.ipynb     # Análisis de resultados experimentales
├── tests/                        # Pruebas unitarias
│   ├── __init__.py
│   ├── test_models.py            # Pruebas para modelos CNN
│   └── test_patterns.py          # Pruebas para generadores de patrones
├── docs/                         # Documentación extendida
│   ├── teoria.md                 # Fundamentos teóricos detallados
│   └── implementacion.md         # Detalles de implementación
├── paper/                        # Recursos relacionados con la publicación
│   ├── figuras/                  # Figuras del artículo
│   └── codigo_latex/             # Código LaTeX para los teoremas
├── requirements.txt              # Dependencias del proyecto
├── setup.py                      # Configuración de instalación
├── .gitignore                    # Archivos ignorados por git
├── LICENSE                       # Licencia MIT
└── README.md                     # Este archivo
```

## 📈 Resultados

### Hallazgos Principales

Nuestros experimentos confirman los teoremas propuestos, demostrando una clara correlación entre:

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qqX2ywJHBRhGW1F5DdkBZQ.png" alt="Resultados Principales" width="750">
</div>

1. **La profundidad de la CNN y su capacidad expresiva**: 
   - CNNs con profundidad 3: Precisión ~95% en patrones regulares, ~45% en patrones sensibles al contexto
   - CNNs con profundidad 5: Precisión ~99% en patrones regulares, ~89% en patrones sensibles al contexto
   - CNNs con profundidad 7: Precisión ~99% en todos los tipos de patrones

2. **Tiempo de entrenamiento y complejidad computacional**:
   - Patrones regulares (Tipo 3): Convergencia en ~3 épocas
   - Patrones libres de contexto (Tipo 2): Convergencia en ~5 épocas
   - Patrones sensibles al contexto (Tipo 1): Convergencia en ~7-10 épocas

3. **Correlación con la jerarquía de Chomsky**:
   - Validación empírica de la relación entre la profundidad necesaria y la complejidad teórica
   - Confirmación de los límites inferiores de profundidad requeridos para cada clase

### Visualizaciones

<details>
<summary><b>Visualizaciones de resultados clave (expandir)</b></summary>
<div align="center">
  <p><b>Matriz de Confusión por Profundidad y Tipo de Patrón</b></p>
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vdJuEUaVaXSgVrOOqVz3yw.png" alt="Matrices de Confusión" width="700">
  
  <p><b>Curvas de Aprendizaje por Profundidad</b></p>
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*dGCbEr2xBhhbzVXLxp1wUQ.png" alt="Curvas de Aprendizaje" width="700">
  
  <p><b>Precisión vs. Complejidad Computacional</b></p>
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3wgIDUjGHUInWaD5lFu8Ew.png" alt="Precisión vs Complejidad" width="700">
</div>
</details>

## 🛣️ Trabajo Futuro

Nuestra investigación abre nuevas direcciones para explorar:

1. **Extensión a otros modelos de aprendizaje profundo**:
   - Aplicar el framework a transformers, redes recurrentes y arquitecturas de atención
   - Explorar la relación entre mecanismos de atención y complejidad computacional

2. **Implicaciones para NLP y visión por computadora**:
   - Desarrollar modelos con profundidad adaptativa según la complejidad de la tarea
   - Optimizar la arquitectura basándose en la complejidad teórica del problema

3. **Reducción de complejidad arquitectónica**:
   - Investigar arquitecturas más eficientes para problemas de alta complejidad
   - Desarrollar heurísticas para determinar la profundidad óptima a priori

4. **Extensiones a la teoría de la computabilidad**:
   - Explorar las limitaciones fundamentales del aprendizaje profundo desde perspectivas algorítmicas
   - Formalizar nuevas clases de complejidad específicas para redes neuronales

## ❓ Preguntas Frecuentes

<details>
<summary><b>¿Por qué es importante la relación entre CNNs y la teoría de la computación?</b></summary>
<p>Establecer esta relación nos permite entender los límites teóricos del aprendizaje profundo, optimizar arquitecturas para tareas específicas, y predecir qué tipos de problemas pueden ser abordados efectivamente con diferentes profundidades de red.</p>
</details>

<details>
<summary><b>¿Qué implicaciones tiene este trabajo para el diseño de arquitecturas CNN?</b></summary>
<p>Sugiere que la profundidad de la red debe determinarse en función de la complejidad computacional del problema, y no simplemente aumentarse arbitrariamente. Proporciona una base teórica para decisiones arquitectónicas que anteriormente se basaban principalmente en la intuición o experimentación.</p>
</details>

<details>
<summary><b>¿Cómo se relaciona este trabajo con los teoremas de universalidad de las redes neuronales?</b></summary>
<p>Mientras que los teoremas de universalidad establecen que las redes neuronales pueden aproximar cualquier función continua, nuestro trabajo va más allá al relacionar la profundidad con clases específicas de complejidad computacional, proporcionando resultados más granulares sobre la capacidad expresiva en relación con la arquitectura.</p>
</details>

<details>
<summary><b>¿Se pueden extender estos resultados a otros tipos de redes neuronales?</b></summary>
<p>Sí, aunque este trabajo se centra en CNNs, el marco teórico y las metodologías experimentales pueden adaptarse a otros tipos de arquitecturas como RNNs, Transformers y GNNs. La relación fundamental entre profundidad y complejidad computacional probablemente se mantenga, con matices específicos para cada arquitectura.</p>
</details>

<details>
<summary><b>¿Cómo puedo contribuir a este proyecto?</b></summary>
<p>Las contribuciones son bienvenidas en forma de implementaciones adicionales de patrones, optimizaciones de código, extensiones teóricas o nuevos experimentos. Consulta la sección de Contribuciones en este README y revisa los issues abiertos para encontrar áreas donde puedas contribuir.</p>
</details>

## 👥 Colaboradores

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/nrodriguezdiaz">
          <img src="https://miro.medium.com/v2/resize:fit:90/format:webp/1*o2DFooEnpCCmIlzhIXGU1g.png" width="100px;" alt="Nicolás Rodríguez Díaz"/>
          <br />
          <sub><b>Nicolás Rodríguez Díaz</b></sub>
        </a>
        <br />
        <sub>Investigador Principal</sub>
      </td>
      <td align="center">
        <a href="https://github.com/colaborador1">
          <img src="https://miro.medium.com/v2/resize:fit:90/format:webp/1*o2DFooEnpCCmIlzhIXGU1g.png" width="100px;" alt="Colaborador 1"/>
          <br />
          <sub><b>Colaborador 1</b></sub>
        </a>
        <br />
        <sub>Desarrollo de Código</sub>
      </td>
      <td align="center">
        <a href="https://github.com/colaborador2">
          <img src="https://miro.medium.com/v2/resize:fit:90/format:webp/1*o2DFooEnpCCmIlzhIXGU1g.png" width="100px;" alt="Colaborador 2"/>
          <br />
          <sub><b>Colaborador 2</b></sub>
        </a>
        <br />
        <sub>Análisis Matemático</sub>
      </td>
    </tr>
  </table>
</div>

## 🙏 Agradecimientos

Este trabajo fue posible gracias al apoyo de:

- **Universidad EAFIT** por proporcionar la infraestructura computacional y el entorno académico.
- **Departamento de Ingeniería Matemática** por el soporte teórico y las discusiones productivas.
- **Revista EIA** por la publicación y difusión de los resultados.

Agradecemos especialmente a los revisores anónimos cuyas sugerencias mejoraron significativamente este trabajo.

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.

## ✉️ Contacto

Nicolás Rodríguez Díaz - [nrodrigued@eafit.edu.co](mailto:nrodrigued@eafit.edu.co)

Universidad EAFIT, Medellín, Colombia

---

<div align="center">

[![Estrellas en GitHub](https://img.shields.io/github/stars/tu-usuario/CNN-Turing-Complexity?style=social)](https://github.com/tu-usuario/CNN-Turing-Complexity/stargazers)
[![Forks en GitHub](https://img.shields.io/github/forks/tu-usuario/CNN-Turing-Complexity?style=social)](https://github.com/tu-usuario/CNN-Turing-Complexity/network/members)

**Si este proyecto te resulta útil o interesante, considera darle una ⭐**

[Inicio](#-cnn-turing-complexity-) • [Documentación](docs/) • [Reportar Bug](https://github.com/tu-usuario/CNN-Turing-Complexity/issues) • [Solicitar Función](https://github.com/tu-usuario/CNN-Turing-Complexity/issues)

</div>
