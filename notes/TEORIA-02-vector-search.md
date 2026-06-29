# Teoria 02 - Vector Search

Este resumen sirve como cuaderno personal para estudiar el modulo `02-vector-search` y entender que esta pasando antes de resolver el homework.

La idea central del modulo es simple pero poderosa: convertir texto en vectores para poder buscar por significado, no solo por palabras exactas.

## 1. Idea central del modulo

En el modulo 1 hicimos RAG usando busqueda lexica: el buscador encontraba documentos por coincidencia de palabras. Eso funciona bien cuando la pregunta y el documento usan terminos parecidos.

Pero falla cuando el usuario pregunta algo con otras palabras.

Ejemplo:

- Query: "How do I store vectors in PostgreSQL?"
- Documento relevante: una pagina sobre `pgvector`

Un buscador de texto puede no entender que "PostgreSQL" y `pgvector` estan relacionados. Un buscador vectorial si puede capturar esa relacion porque busca cercania semantica.

La pregunta del modulo es:

> Como representamos texto como numeros para poder medir significado?

La respuesta es:

> Con embeddings.

## 2. Que es un embedding

Un embedding es un vector numerico que representa el significado de un texto.

En este homework se usa `all-MiniLM-L6-v2`, que genera vectores de 384 dimensiones. Eso significa que cada texto se transforma en una lista de 384 numeros.

Ejemplo mental:

```python
text = "How does approximate nearest neighbor search work?"
v = embedder.encode(text)
len(v)  # 384
```

No importa mucho interpretar cada dimension individual. Lo importante es que textos parecidos terminan cerca en el espacio vectorial.

### Intuicion

Si dos frases hablan de lo mismo, sus vectores apuntan en direcciones parecidas.

- "store vectors in PostgreSQL"
- "use pgvector for embeddings"

Aunque no comparten todas las palabras, semanticamente estan cerca.

## 3. Distancia, similitud y cosine similarity

Una vez que tenemos vectores, necesitamos medir que tan parecidos son.

La metrica mas comun en embeddings de texto es cosine similarity.

### Que mide

Mide el angulo entre dos vectores, no tanto su longitud.

- Valor cercano a `1`: muy parecidos.
- Valor cercano a `0`: poca relacion.
- Valor negativo: direcciones opuestas o relacion debil.

En el homework, el embedder devuelve vectores normalizados. Eso simplifica todo:

```python
similarity = vector_a.dot(vector_b)
```

Si los vectores estan normalizados, el producto punto ya es cosine similarity.

## 4. Busqueda vectorial

La busqueda vectorial tiene este flujo:

1. Convertir todos los documentos en embeddings.
2. Guardar esos embeddings en una matriz o indice.
3. Convertir la query del usuario en otro embedding.
4. Calcular similitud entre la query y cada documento.
5. Ordenar por similitud.
6. Devolver los documentos mas cercanos.

En version minima con numpy:

```python
query_vector = embedder.encode(query)
scores = X.dot(query_vector)
best_idx = scores.argmax()
best_doc = documents[best_idx]
```

Donde `X` es una matriz con un vector por documento o chunk.

## 5. Por que usamos chunks

Una pagina completa puede hablar de muchas cosas. Si embeddeas una pagina entera, el vector resultante mezcla varios temas.

Eso diluye el significado.

Ejemplo:

- Una leccion puede hablar de SQLite, embeddings, minsearch y ANN.
- La query pregunta solo por approximate nearest neighbor search.
- Si el documento completo es muy amplio, su embedding puede no ser tan preciso.

La solucion es dividir documentos largos en fragmentos mas chicos.

En el homework:

```python
chunks = chunk_documents(documents, size=2000, step=1000)
```

Esto crea ventanas de 2000 caracteres con solapamiento de 1000 caracteres.

### Por que hay solapamiento

Si cortas un documento justo en medio de una explicacion, podes perder contexto. El solapamiento hace que una idea que queda entre dos cortes aparezca completa en al menos un chunk.

### Tradeoff

Chunks mas chicos:

- mejor precision local,
- menos contexto irrelevante,
- mas cantidad de vectores.

Chunks mas grandes:

- mas contexto por resultado,
- menos vectores,
- mas ruido semantico.

## 6. Approximate nearest neighbor search

La busqueda vectorial ingenua compara la query contra todos los vectores.

Eso esta bien si tenes pocos documentos:

```python
scores = X.dot(query_vector)
```

Pero si tenes millones de vectores, comparar contra todos puede ser caro.

Approximate nearest neighbor search, o ANN, busca resultados casi iguales a los mejores, pero mucho mas rapido.

La idea es aceptar una pequena perdida de exactitud a cambio de velocidad.

### Busqueda exacta

- Compara contra todos los vectores.
- Encuentra el verdadero top K.
- Puede ser lenta a gran escala.

### ANN

- Usa estructuras especiales para reducir el espacio de busqueda.
- Encuentra vecinos muy buenos, aunque no siempre perfectos.
- Es lo que usan muchas bases vectoriales en produccion.

## 7. Minsearch para vector search

En el modulo 1 usamos `minsearch.Index` para busqueda lexica.

En este modulo aparece `minsearch.VectorSearch`.

La diferencia:

```python
from minsearch import VectorSearch

vector_search = VectorSearch(keyword_fields={"filename"})
vector_search.fit(vectors, chunks)
results = vector_search.search(query_vector, num_results=5)
```

### Que guarda

`VectorSearch` guarda:

- los vectores,
- el payload asociado a cada vector,
- campos keyword como `filename`.

El payload suele ser el chunk original:

```python
{
    "filename": "...",
    "content": "...",
    "start": 1000
}
```

Eso permite recuperar no solo el score, sino tambien el texto y la fuente.

## 8. Text search vs vector search

Este modulo no dice que vector search reemplace siempre a keyword search. Dice algo mas interesante:

> Cada enfoque falla de manera distinta.

### Text search

Busca coincidencias de terminos.

Ventajas:

- Muy bueno para nombres exactos.
- Bueno para codigos, IDs, comandos y palabras raras.
- Predecible.

Debilidades:

- No entiende sinonimos.
- No entiende parafrasis.
- Puede fallar si la query usa otras palabras.

### Vector search

Busca por significado.

Ventajas:

- Encuentra conceptos relacionados aunque no usen las mismas palabras.
- Tolera mejor preguntas naturales.
- Sirve muy bien para semantica.

Debilidades:

- Puede perder terminos exactos importantes.
- Puede traer resultados semanticamente cercanos pero no exactos.
- Depende de la calidad del modelo de embeddings.

## 9. Hybrid search

Hybrid search combina busqueda lexica y busqueda vectorial.

La intuicion:

> Si text search y vector search tienen fortalezas distintas, podemos usar las dos y fusionar resultados.

En el homework se hace asi:

1. Buscar con vector search.
2. Buscar con text search.
3. Combinar rankings con RRF.

## 10. Reciprocal Rank Fusion, RRF

RRF es una forma simple de fusionar rankings.

No usa los scores crudos porque los scores de text search y vector search no viven en la misma escala.

En vez de eso, usa posiciones.

Formula:

```text
RRF(d) = sum over lists of 1 / (k + rank(d))
```

Con `k = 60`.

### Intuicion

Un documento que aparece alto en ambas listas deberia subir.

Un documento que aparece primero en una lista pero no aparece en la otra puede seguir siendo bueno, pero no recibe el refuerzo doble.

### Implementacion del homework

```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
```

La clave del documento es `(filename, start)` porque puede haber varios chunks del mismo archivo.

## 11. ONNX embedder

En las lecciones se puede usar `sentence-transformers`, pero en el homework se usa un embedder ONNX.

### Por que ONNX

ONNX Runtime permite correr el modelo sin instalar PyTorch.

Ventajas:

- Instalacion mas liviana.
- No necesita CUDA.
- Corre bien en maquinas simples.
- Es mas facil de usar en entornos chicos como Codespaces.

### Archivos importantes

En HW2 aparecen:

- `download.py`: descarga el modelo desde HuggingFace.
- `embedder.py`: define la clase `Embedder`.

El modelo se guarda en:

```text
models/Xenova/all-MiniLM-L6-v2/
```

Y se usa asi:

```python
from embedder import Embedder

embedder = Embedder("models/Xenova/all-MiniLM-L6-v2")
v = embedder.encode("some text")
vectors = embedder.encode_batch(texts)
```

## 12. Vector databases

El modulo tambien conecta la idea teorica con herramientas reales para guardar y buscar vectores.

### SQLite vector search

Sirve para entender una version local y simple.

La idea:

- guardar documentos,
- guardar embeddings,
- hacer consultas por similitud,
- tener algo liviano sin montar infraestructura grande.

### PostgreSQL con pgvector

`pgvector` agrega soporte vectorial a PostgreSQL.

Es importante porque muchas aplicaciones ya usan PostgreSQL como base principal. Si agregas `pgvector`, podes guardar embeddings y hacer busqueda semantica sin introducir otra base especializada.

Ejemplo mental:

```sql
CREATE EXTENSION vector;
```

Luego guardas una columna vectorial y consultas por distancia.

### Cuando usar una vector DB dedicada

Si el volumen crece mucho o necesitas ANN avanzado, filtros complejos, replicas o latencia baja a gran escala, puede convenir una base vectorial dedicada.

Pero para aprender y para muchos prototipos, numpy, minsearch, SQLite o PostgreSQL con pgvector alcanzan perfectamente.

## 13. Relacion con RAG

Este modulo se enfoca solo en search, no en generar respuestas.

Pero es una pieza central de RAG.

En RAG, la calidad final depende mucho de la recuperacion:

```text
pregunta -> search -> contexto -> LLM -> respuesta
```

Si el search trae malos documentos, el LLM recibe mal contexto. Despues puede escribir bonito, pero va a estar parado sobre datos flojos.

Por eso este modulo se concentra en mejorar retrieval antes de volver a RAG.

## 14. Flujo paso a paso recomendado para estudiar el modulo

Si queres estudiar sin perderte, este orden funciona:

1. Entender que un embedding es una representacion numerica del significado.
2. Probar `embedder.encode()` con una query corta y ver que devuelve 384 numeros.
3. Leer sobre cosine similarity y entender por que `dot()` alcanza cuando los vectores estan normalizados.
4. Cargar las 72 lecciones del curso con `gitsource`.
5. Embeddear una pagina completa y comparar similitud con la query.
6. Aplicar chunking para mejorar precision.
7. Crear la matriz `X` con todos los embeddings de chunks.
8. Hacer busqueda manual con `X.dot(v)`.
9. Repetir lo mismo con `minsearch.VectorSearch`.
10. Comparar con `minsearch.Index` para ver diferencias entre vector y keyword search.
11. Fusionar resultados con RRF.
12. Pensar que enfoque conviene segun el problema: texto, vector o hybrid.

## 15. Que mirar para resolver el homework

### Q1

Embeddear la query:

```text
How does approximate nearest neighbor search work?
```

Leer el primer valor del vector.

Concepto clave:

- Un texto se transforma en un vector de 384 numeros.

### Q2

Embeddear una pagina concreta y comparar con la query de Q1.

Concepto clave:

- Como los vectores estan normalizados, `dot()` equivale a cosine similarity.

### Q3

Chunkear documentos, embeddear chunks y buscar a mano con numpy.

Conceptos clave:

- Las paginas completas pueden ser demasiado amplias.
- Los chunks hacen la busqueda mas precisa.

### Q4

Usar `VectorSearch` de minsearch.

Concepto clave:

- En la practica no queres escribir siempre la busqueda manual; usas una libreria o base vectorial.

### Q5

Comparar resultados entre:

- `VectorSearch`
- `Index`

Concepto clave:

- Vector search encuentra significado.
- Text search encuentra palabras.

### Q6

Combinar resultados con RRF.

Concepto clave:

- Hybrid search suele ser mas robusto porque aprovecha dos senales distintas.

## 16. Comparacion rapida

### Text search

- Mejor para coincidencias exactas.
- Mas facil de interpretar.
- Puede fallar con sinonimos o parafrasis.

### Vector search

- Mejor para significado.
- Mas flexible con lenguaje natural.
- Puede fallar con terminos exactos.

### Hybrid search

- Combina ambas senales.
- Suele ser una buena opcion por defecto.
- Requiere decidir como fusionar resultados.

## 17. Errores comunes

### Confundir embeddings con palabras clave

Un embedding no guarda "palabras importantes" de forma directa. Guarda una representacion numerica aprendida por el modelo.

### Pensar que vector search siempre gana

No siempre. Si buscas un ID, un comando exacto o un nombre de archivo, keyword search puede ser mejor.

### Comparar scores incompatibles

No conviene mezclar score de BM25/text search con cosine similarity directamente. Por eso RRF usa rankings.

### Embedear documentos demasiado largos

Un documento largo puede mezclar temas. Chunking suele mejorar retrieval.

### Olvidar normalizacion

Si los vectores no estan normalizados, `dot()` no es exactamente cosine similarity. En este homework el embedder ya normaliza por defecto.

## 18. Regla practica para elegir busqueda

Si la query depende de palabras exactas:

- usa text search.

Si la query depende de significado:

- usa vector search.

Si no sabes, o queres robustez:

- usa hybrid search.

En sistemas reales, lo correcto es medir. Este modulo ensena las tecnicas; el modulo de evaluation ensena como comparar formalmente cual funciona mejor.

## 19. Resumen mental del modulo

Este modulo agrega la pieza semantica al RAG.

En el modulo 1 aprendiste:

```text
buscar texto -> armar contexto -> llamar LLM
```

En el modulo 2 aprendes:

```text
texto -> embedding -> vector search -> mejores chunks
```

Y despues:

```text
keyword search + vector search -> RRF -> hybrid search
```

La gran idea:

> La calidad de un sistema RAG empieza antes del LLM: empieza en la calidad de la busqueda.

