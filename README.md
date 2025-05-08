#!/bin/bash
# Configuración completa del repositorio Git para el proyecto CNN-Turing-Complexity

# Paso 1: Crear el directorio del proyecto y navegar a él
mkdir CNN-Turing-Complexity
cd CNN-Turing-Complexity

# Paso 2: Inicializar el repositorio git
git init

# Paso 3: Configurar la estructura básica
mkdir -p src/{models,data,training,visualization} scripts notebooks tests

# Paso 4: Crear archivo .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pt
*.pth

# Resultados de experimentos
resultados/
*.log

# Entorno Virtual
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Específicos del sistema operativo
.DS_Store
Thumbs.db
EOF

# Paso 5: Crear README.md
cat > README.md << 'EOF'
# CNN-Turing-Complexity

Explorando la relación entre la profundidad de redes CNN y la complejidad computacional.

## Descripción General

Este proyecto implementa experimentos que investigan cómo la profundidad de las redes neuronales convolucionales se relaciona con el reconocimiento de patrones de diferentes clases de complejidad computacional (basadas en la jerarquía de Chomsky).

Este código implementa la investigación descrita en "Fundamentos Matemáticos Aplicados a Sistemas De Redes Neuronales Convolucionales y las Máquinas de Turing" publicado en la revista EIA.

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/CNN-Turing-Complexity.git
cd CNN-Turing-Complexity

# Crear y activar un entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar experimento completo
python scripts/experiment.py --modo completo --num_epocas 30

# Ejecutar experimento simplificado
python scripts/experiment.py --modo simplificado --num_epocas 10
```

## Estructura del Proyecto

- `src/`: Código fuente principal
  - `models/`: Implementación de arquitecturas CNN
  - `data/`: Generadores de patrones y gestión de datos
  - `training/`: Funcionalidades de entrenamiento y evaluación
  - `visualization/`: Herramientas de visualización
- `scripts/`: Scripts ejecutables
- `notebooks/`: Jupyter notebooks para análisis
- `tests/`: Pruebas unitarias

## Licencia

Este proyecto está bajo la Licencia MIT.
EOF

# Paso 6: Crear requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.20.0
matplotlib>=3.4.0
torch>=1.10.0
torchvision>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
seaborn>=0.11.0
pandas>=1.3.0
EOF

# Paso 7: Copiar el archivo principal
# Asumiendo que el archivo corazon.py está en el directorio actual
cp corazon.py scripts/experiment.py

# Paso 8: Crear archivos de módulos Python básicos
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/visualization/__init__.py
touch tests/__init__.py

# Paso 9: Extraer las clases principales a sus propios archivos
# Por ejemplo, extraer la clase GeneradorPatrones a su propio archivo

cat > src/data/generador_patrones.py << 'EOF'
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GeneradorPatrones:
    """
    Clase para generar diferentes tipos de patrones visuales que representan
    distintos niveles de complejidad computacional según la jerarquía de Chomsky.
    """
    
    @staticmethod
    def crear_sierpinski(orden, tamano=64):
        """
        Genera un triángulo de Sierpinski de un orden específico.
        
        Args:
            orden (int): La profundidad de recursión del triángulo de Sierpinski
            tamano (int): El tamaño de la imagen de salida
            
        Returns:
            np.ndarray: Un array 2D que representa el triángulo de Sierpinski
        """
        def recursion(x, y, tamano, orden):
            if orden == 0:
                imagen[y:y+tamano, x:x+tamano] = 1
                return
            
            nuevo_tamano = tamano // 2
            
            recursion(x, y, nuevo_tamano, orden-1)  # Esquina superior izquierda
            recursion(x + nuevo_tamano, y, nuevo_tamano, orden-1)  # Esquina superior derecha
            recursion(x + nuevo_tamano//2, y + nuevo_tamano, nuevo_tamano, orden-1)  # Esquina inferior
        
        imagen = np.zeros((tamano, tamano), dtype=np.float32)
        recursion(0, 0, tamano//2, orden)
        return imagen

    @staticmethod
    def crear_patron_parentesis(profundidad, tamano=64):
        """
        Crea un patrón visual basado en paréntesis balanceados.
        
        Args:
            profundidad (int): La profundidad de anidamiento de los paréntesis
            tamano (int): El tamaño de la imagen de salida
            
        Returns:
            np.ndarray: Un array 2D que representa el patrón de paréntesis
        """
        imagen = np.zeros((tamano, tamano), dtype=np.float32)
        
        # Genera una secuencia de paréntesis balanceados
        def generar_parentesis_balanceados(n):
            if n == 0:
                return [""]
            resultado = []
            for i in range(n):
                for izquierda in generar_parentesis_balanceados(i):
                    for derecha in generar_parentesis_balanceados(n-i-1):
                        resultado.append("(" + izquierda + ")" + derecha)
            return resultado
        
        # Limitar la profundidad para evitar explosión combinatoria
        profundidad = min(profundidad, 4)
        
        # Usar la primera secuencia generada
        secuencia = generar_parentesis_balanceados(profundidad)[0]
        
        # Visualizar la secuencia como un patrón
        x, y = tamano // 4, tamano // 2
        grosor = 2
        for caracter in secuencia:
            if caracter == '(':
                for i in range(grosor):
                    for j in range(tamano // 8):
                        if x + j < tamano and y - j >= 0:
                            imagen[y - j, x + i] = 1
                x += grosor
            else:  # caracter == ')'
                for i in range(grosor):
                    for j in range(tamano // 8):
                        if x + i < tamano and y + j < tamano:
                            imagen[y + j, x + i] = 1
                x += grosor
                
        return imagen

    @staticmethod
    def crear_patron_sensible_contexto(n, tamano=64):
        """
        Crea un patrón visual basado en la gramática sensible al contexto a^n b^n c^n.
        
        Args:
            n (int): El parámetro n en a^n b^n c^n
            tamano (int): El tamaño de la imagen de salida
            
        Returns:
            np.ndarray: Un array 2D que representa el patrón sensible al contexto
        """
        imagen = np.zeros((tamano, tamano), dtype=np.float32)
        
        # Limitar n para mantener el patrón visible
        n = min(n, 10)
        
        # Crear patrón con:
        # - n barras horizontales en la sección superior (a^n)
        # - n barras verticales en la sección media (b^n)
        # - n barras diagonales en la sección inferior (c^n)
        
        altura_seccion = tamano // 3
        ancho_barra = max(2, tamano // (n * 2))
        espaciado = max(4, tamano // (n + 1))
        
        # Sección a^n (barras horizontales)
        for i in range(n):
            y = (i + 1) * espaciado
            if y + ancho_barra < altura_seccion:
                imagen[y:y+ancho_barra, tamano//4:3*tamano//4] = 1
        
        # Sección b^n (barras verticales)
        for i in range(n):
            x = tamano // 4 + i * espaciado
            if x + ancho_barra < 3 * tamano // 4:
                imagen[altura_seccion:2*altura_seccion, x:x+ancho_barra] = 1
        
        # Sección c^n (barras diagonales)
        for i in range(n):
            for j in range(altura_seccion):
                x = tamano // 4 + i * espaciado + j // 2
                y = 2 * altura_seccion + j
                if x + ancho_barra < 3 * tamano // 4 and y + ancho_barra < tamano:
                    imagen[y:y+ancho_barra, x:x+ancho_barra] = 1
        
        return imagen

    @staticmethod
    def crear_patron_regular(tipo_patron, tamano=64):
        """
        Crea patrones regulares simples (no recursivos).
        
        Args:
            tipo_patron (int): El tipo de patrón a generar (0-2)
            tamano (int): El tamaño de la imagen de salida
            
        Returns:
            np.ndarray: Un array 2D que representa el patrón regular
        """
        imagen = np.zeros((tamano, tamano), dtype=np.float32)
        
        if tipo_patron == 0:  # Cuadrícula
            for i in range(0, tamano, tamano // 8):
                imagen[i:i+2, :] = 1
                imagen[:, i:i+2] = 1
        
        elif tipo_patron == 1:  # Círculos concéntricos
            centro = tamano // 2
            for r in range(0, tamano // 2, tamano // 16):
                for i in range(tamano):
                    for j in range(tamano):
                        if abs((i-centro)**2 + (j-centro)**2 - r**2) < tamano:
                            imagen[i, j] = 1
        
        elif tipo_patron == 2:  # Patrón diagonal
            for i in range(tamano):
                imagen[i, i] = 1
                imagen[i, tamano-i-1] = 1
        
        return imagen
    
    @staticmethod
    def crear_automata_limitado_lineal(regla, tamano=64):
        """
        Crea un patrón basado en el comportamiento de un autómata limitado lineal (LBA).
        
        Args:
            regla (int): Determina la regla LBA a usar (0-2)
            tamano (int): El tamaño de la imagen de salida
            
        Returns:
            np.ndarray: Un array 2D que representa el patrón LBA
        """
        imagen = np.zeros((tamano, tamano), dtype=np.float32)
        
        # Inicializar la primera fila con un patrón simple
        if regla == 0:
            # Una sola celda en el medio
            imagen[0, tamano//2] = 1
        elif regla == 1:
            # Tres celdas en el medio
            imagen[0, tamano//2-1:tamano//2+2] = 1
        else:
            # Patrón alternante
            imagen[0, ::2] = 1
        
        # Aplicar regla para generar filas subsecuentes
        for i in range(1, tamano):
            for j in range(1, tamano-1):
                # Regla similar al autómata celular Regla 110 pero limitado
                vecindario = (imagen[i-1, j-1], imagen[i-1, j], imagen[i-1, j+1])
                
                if regla == 0:
                    # Regla que genera un patrón tipo Sierpinski
                    imagen[i, j] = vecindario[0] ^ vecindario[2]
                elif regla == 1:
                    # Regla que genera un patrón limitado más complejo
                    if vecindario == (1, 1, 1) or vecindario == (1, 0, 0) or vecindario == (0, 1, 0):
                        imagen[i, j] = 1
                else:
                    # Regla que genera un patrón serpenteante
                    if sum(vecindario) >= 2 or (vecindario[0] == 1 and vecindario[2] == 1):
                        imagen[i, j] = 1
        
        return imagen
EOF

cat > src/models/cnn_moderna.py << 'EOF'
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class BloqueResidual(nn.Module):
    """
    Un bloque residual con conexión de atajo para mejor flujo de gradiente.
    """
    
    def __init__(self, capas, reduccion_identidad=None):
        """
        Inicializa el bloque residual.
        
        Args:
            capas (nn.Sequential): Capas convolucionales en el bloque
            reduccion_identidad (nn.Module, optional): Reduce la conexión de identidad
        """
        super(BloqueResidual, self).__init__()
        self.capas = capas
        self.reduccion_identidad = reduccion_identidad
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Paso hacia adelante con conexión de atajo"""
        identidad = x
        
        out = self.capas(x)
        
        # Aplicar reducción de identidad si es necesario
        if self.reduccion_identidad is not None:
            identidad = self.reduccion_identidad(x)
        
        out += identidad
        out = self.relu(out)
        
        return out

class CNNModerna(nn.Module):
    """
    Una arquitectura CNN mejorada con componentes modernos como conexiones residuales
    y normalización por lotes para mejor rendimiento y convergencia.
    """
    
    def __init__(self, profundidad, num_clases=4, forma_entrada=(1, 64, 64)):
        """
        Inicializa el modelo CNN.
        
        Args:
            profundidad (int): Parámetro de profundidad que controla la complejidad del modelo
            num_clases (int): Número de clases a clasificar
            forma_entrada (tuple): Forma de las imágenes de entrada (C, H, W)
        """
        super(CNNModerna, self).__init__()
        self.profundidad = profundidad
        self.forma_entrada = forma_entrada
        
        # Determinar arquitectura basada en la profundidad
        num_bloques_conv = max(2, min(profundidad, 10))  # Al menos 2, máximo 10 bloques
        
        # Seguir las dimensiones actuales
        canales_entrada = forma_entrada[0]
        altura_actual, ancho_actual = forma_entrada[1], forma_entrada[2]
        
        # Crear capas convolucionales con conexiones residuales
        self.bloques_conv = nn.ModuleList()
        
        # Características iniciales
        canales_salida = 16
        
        # Convolución inicial
        self.conv_inicial = nn.Sequential(
            nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True)
        )
        
        # Bloques convolucionales
        for i in range(num_bloques_conv):
            # Crear bloque residual
            bloque = self._hacer_bloque_residual(canales_salida, canales_salida * 2 if i < 3 else canales_salida)
            self.bloques_conv.append(bloque)
            
            # Reducir dimensiones después de cada dos bloques o en el último bloque
            if i % 2 == 1 or i == num_bloques_conv - 1:
                self.bloques_conv.append(nn.Sequential(
                    nn.Conv2d(canales_salida * 2 if i < 3 else canales_salida, 
                             canales_salida * 2 if i < 3 else canales_salida, 
                             kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(canales_salida * 2 if i < 3 else canales_salida),
                    nn.ReLU(inplace=True)
                ))
                altura_actual //= 2
                ancho_actual //= 2
            
            # Actualizar canales para la siguiente capa
            if i < 3:
                canales_salida *= 2
        
        # Número final de canales
        self.canales_finales = canales_salida
        
        # Calcular tamaño de características para capa completamente conectada
        self.tamano_caracteristicas = self.canales_finales * altura_actual * ancho_actual
        
        # Capas completamente conectadas con dropout para regularización
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
        
        # Inicializar pesos
        self._inicializar_pesos()
        
        logger.info(f"CNN creada con {num_bloques_conv} bloques conv, forma de salida: {canales_salida}×{altura_actual}×{ancho_actual}")
        logger.info(f"Tamaño de características: {self.tamano_caracteristicas}, parámetros: {self._contar_parametros():,}")
    
    def _hacer_bloque_residual(self, canales_entrada, canales_salida):
        """
        Crea un bloque residual con conexión de atajo.
        """
        # Si las dimensiones cambian, necesitamos un atajo de proyección
        reduccion_identidad = None
        if canales_entrada != canales_salida:
            reduccion_identidad = nn.Sequential(
                nn.Conv2d(canales_entrada, canales_salida, kernel_size=1),
                nn.BatchNorm2d(canales_salida)
            )
        
        capas = nn.Sequential(
            # Primera convolución
            nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
            # Segunda convolución
            nn.Conv2d(canales_salida, canales_salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_salida),
            # No hay ReLU aquí - se aplica después de la conexión de atajo
        )
        
        # Crear el bloque residual
        return BloqueResidual(capas, reduccion_identidad)
    
    def _inicializar_pesos(self):
        """Inicializar pesos usando inicialización He"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def _contar_parametros(self):
        """Contar el número de parámetros entrenables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Paso hacia adelante a través de la red"""
        x = self.conv_inicial(x)
        
        # Pasar a través de bloques convolucionales
        for bloque in self.bloques_conv:
            x = bloque(x)
        
        # Pasar a través de capas completamente conectadas
        x = self.capas_fc(x)
        
        return x
EOF

# Paso 10: Realizar primer commit
git add .
git commit -m "Configuración inicial del proyecto CNN-Turing-Complexity"

# Paso 11: Configurar un repositorio remoto (opcional)
# git remote add origin https://github.com/tu-usuario/CNN-Turing-Complexity.git
# git push -u origin master

echo "¡Repositorio Git configurado exitosamente!"
echo "Estructura del proyecto:"
find . -type f | grep -v "__pycache__" | sort

echo "Para completar la configuración del repositorio remoto, ejecuta:"
echo "git remote add origin https://github.com/tu-usuario/CNN-Turing-Complexity.git"
echo "git push -u origin master"
