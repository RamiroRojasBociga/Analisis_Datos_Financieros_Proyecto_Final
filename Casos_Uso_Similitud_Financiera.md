# 📈 Casos de Uso Práctico: Algoritmos de Similitud en Finanzas

Este documento es una guía complementaria diseñada para la sustentación del proyecto. Su objetivo es explicar de forma clara y práctica **para qué sirven** los algoritmos de similitud en el mundo real de las finanzas y el trading algorítmico.

---

## 1. Correlación de Pearson: Diversificación de Portafolios (Gestión de Riesgo)

### ¿Qué mide en términos simples?
Mide si dos acciones se mueven en la misma dirección (Correlación positiva cercana a 1), si se mueven en direcciones opuestas (Correlación negativa cercana a -1), o si no tienen relación alguna (Correlación 0).

### Aplicación Financiera: Construcción de Portafolios Seguros
Los fondos de inversión utilizan Pearson para evitar el riesgo sistémico. La regla de oro en finanzas es: *"No pongas todos los huevos en la misma canasta"*. Si compras acciones que están altamente correlacionadas, tu portafolio es muy frágil frente a caídas del sector.

### Ejemplo Práctico en el Proyecto
Si el algoritmo detecta que **Ecopetrol** y **Chevron** tienen una correlación de Pearson de **0.95**, significa que se mueven casi en paralelo (dependen ambas del precio del petróleo crudo). 
Si un inversor compra ambas, no está diversificando su riesgo. Para crear un portafolio robusto, el algoritmo le sugeriría mezclar **Ecopetrol** (petróleo) con **Amazon** (tecnología), cuya correlación podría ser cercana a **0.1** (independientes). Si el petróleo cae, las acciones de Amazon amortiguan la pérdida del portafolio.

---

## 2. Distancia Euclidiana y Similitud del Coseno: Pairs Trading (Arbitraje)

### ¿Qué miden en términos simples?
* **Distancia Euclidiana:** Mide la distancia absoluta en precio/magnitud entre dos acciones en un gráfico a lo largo del tiempo.
* **Similitud del Coseno:** Evalúa únicamente la "direccionalidad" del movimiento, ignorando si una acción vale $100 y la otra $10. Si ambas suben un 2%, el coseno es altísimo.

### Aplicación Financiera: Estrategia "Pairs Trading" (Operación de Pares)
Los fondos de cobertura (Hedge Funds) buscan dos acciones que históricamente han sido "gemelas" (alta similitud de coseno y baja distancia euclidiana). Si en algún momento estas acciones gemelas se separan abruptamente, el analista asume que es un error temporal del mercado y que pronto volverán a su equilibrio histórico.

### Ejemplo Práctico en el Proyecto
Imagina que el algoritmo determina que **Coca-Cola** y **Pepsi** tienen una Similitud del Coseno de **0.98** (casi idénticas direccionalmente). 
Hoy, la Distancia Euclidiana entre ambas se dispara drásticamente porque Coca-Cola subió un 5% y Pepsi bajó un 4% por un rumor pasajero. 
* **La estrategia algorítmica:** El analista vendería acciones de Coca-Cola (porque están temporalmente sobrevaloradas) y compraría acciones de Pepsi (porque están infravaloradas). Cuando la anomalía pase y vuelvan a moverse juntas, el inversor ganará dinero en ambas operaciones.

---

## 3. Dynamic Time Warping (DTW): Predicción por Patrones Desfasados

### ¿Qué mide en términos simples?
A diferencia de la Distancia Euclidiana que compara el "Día 1" estrictamente con el "Día 1" (alineación rígida), el DTW es "elástico". Es capaz de alinear la forma de dos curvas aunque hayan ocurrido a diferentes velocidades o en diferentes momentos del tiempo.

### Aplicación Financiera: Detección Temprana de Crisis y Cisnes Negros
Los mercados financieros son cíclicos; la historia a menudo se repite, pero no exactamente al mismo ritmo. DTW se usa para buscar si el comportamiento de una acción hoy se parece a un colapso financiero del pasado.

### Ejemplo Práctico en el Proyecto
Supongamos que estamos analizando el índice **S&P 500** en la actualidad. Las gráficas muestran una leve caída seguida de un estancamiento.
Aplicando el algoritmo DTW contra bases de datos históricas de crisis pasadas (Ej: Crisis de las PuntoCom en el año 2000 o la Burbuja Inmobiliaria del 2008). 
El algoritmo detecta que la forma geométrica de la caída actual tiene una distancia de costo mínimo (DTW) casi nula en comparación con los 2 meses previos al estallido del 2008. 
Aunque en 2008 el mercado tardó 60 días en formar el patrón, y ahora solo ha tardado 20 días, el DTW reconoce la **misma forma desfasada en el tiempo**. Esto genera una alerta algorítmica temprana para liquidar posiciones antes de que suceda el inminente "crash" que el algoritmo ha identificado en la historia.

---
**Conclusión:** La inclusión de estos algoritmos en el proyecto no es un simple ejercicio matemático, sino que replica la infraestructura de software predictivo y de control de riesgo que utilizan firmas de Wall Street como Renaissance Technologies o BlackRock.
