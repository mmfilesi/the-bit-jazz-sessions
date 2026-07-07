"""
Este script identifica qué entidades del grafo se mencionan en la pregunta, recupera su vecindario a 1-2 saltos desde Neo4j, y le pasa ese subgrafo como contexto a DeepSeek para que redacte la respuesta final.

Por qué no hace falta la Fase 4 (comunidades) para esto: Local Search solo necesita recorrer el grafo desde un punto de partida conocido, no requiere resúmenes de alto nivel de todo el grafo, así que puede ejecutarse con lo que ya teníamos tras la Fase 3.

El pipeline completo de este script tiene 4 pasos: (1) identificar las entidades ancla mencionadas en la pregunta, (2) recuperar su vecindario desde Neo4j, (3) convertir ese subgrafo en texto legible, y (4) pasarle ese texto como contexto a un segundo LLM que redacta la respuesta final. Es literalmente el patrón "Retrieval-Augmented Generation" aplicado a un grafo en vez de a texto plano: primero se recupera contexto relevante, después se genera la respuesta apoyándose en ese contexto.
"""

import json
import os
import unicodedata

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# La pregunta se deja fija como constante en vez de pedirla por línea de comandos, para mantener el script simple en esta primera pasada del tutorial. Cambiarla a mano es suficiente para experimentar con otras preguntas locales.
QUESTION = "¿Cómo se conocieron Jon Snow y Daenerys?"

# Cuántos saltos de distancia recorremos desde cada entidad ancla. Lo dejamos como constante configurable arriba del todo para que sea fácil experimentar subiendo o bajando el radio de búsqueda y observar cómo cambia el contexto recuperado y, en consecuencia, la respuesta final.
MAX_HOPS = 2


def normalize_name(name: str) -> str:
    """Misma normalización que usamos al cargar el grafo en la Fase 3 (minúsculas, sin tildes). Es imprescindible reutilizar exactamente la misma lógica aquí, porque los nodos en Neo4j se guardaron con esa clave normalizada, y si buscáramos con una normalización distinta, simplemente no encontraríamos coincidencias."""
    text = name.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


# Este prompt hace de "extractor de entidades" ligero: en vez de intentar detectar nombres propios con reglas de texto (mayúsculas, expresiones regulares, etc.), delegamos esa tarea en el propio LLM, que entiende de forma mucho más robusta que "Jon" y "Jon Snow" se refieren a la misma clase de cosa (un nombre propio a buscar en el grafo), incluso si la pregunta usa una forma abreviada o coloquial.
ENTITY_EXTRACTION_SYSTEM_PROMPT = """Identifica los nombres propios de personajes, casas nobles, organizaciones, regiones o eventos que se mencionan en la pregunta del usuario. Responde ÚNICAMENTE con un JSON de la forma {{"entidades": ["nombre1", "nombre2"]}}, sin texto adicional."""

ENTITY_EXTRACTION_USER_PROMPT = """Pregunta: {question}"""


def extract_anchor_entities(chain, question: str) -> list[str]:
    """Le pide al LLM que identifique los nombres propios mencionados en la pregunta, para usarlos como nodos de partida (nodos ancla) al recorrer el grafo. El resultado de esta función es la lista de nombres que luego intentaremos localizar como nodos reales en Neo4j."""
    response = chain.invoke({"question": question})
    text = response.content.strip()
    # Igual que en la Fase 2, nos protegemos por si el modelo envuelve la respuesta en un bloque de código markdown pese a habérselo pedido explícitamente que no lo haga.
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    return data.get("entidades", [])


def fetch_neighborhood(session, anchor_names: list[str], max_hops: int) -> list[dict]:
    """Recupera, para cada entidad ancla, todas las relaciones alcanzables en hasta max_hops saltos, en cualquier dirección. Devuelve una lista de tripletas (sujeto, relación, objeto, momento) ya en formato simple de diccionario Python, lista para convertir a texto en el siguiente paso, sin que el resto del script tenga que preocuparse de los tipos propios del driver de Neo4j."""
    triples = []
    for name in anchor_names:
        # La sintaxis [*1..max_hops] es la que permite un recorrido de longitud variable en Cypher: en vez de fijar el patrón a exactamente un salto, le decimos a Neo4j que explore caminos de entre 1 y max_hops relaciones de longitud, en cualquier dirección (por eso no ponemos flecha -> ni <- en el patrón).
        result = session.run(
            f"""
            MATCH (anchor {{normalizedName: $normalized_name}})
            MATCH path = (anchor)-[*1..{max_hops}]-(neighbor)
            UNWIND relationships(path) AS rel
            RETURN startNode(rel).nombre AS subject, type(rel) AS relation, endNode(rel).nombre AS object, rel.momento AS moment
            """,
            normalized_name=normalize_name(name),
        )
        # UNWIND relationships(path) descompone cada camino completo (que puede tener 1 o 2 relaciones seguidas) en sus relaciones individuales, para poder leer el sujeto y el objeto de cada tramo por separado, en vez de quedarnos solo con el nodo final del camino.
        for record in result:
            triples.append(dict(record))
    return triples


def deduplicate_triples(triples: list[dict]) -> list[dict]:
    """Varias entidades ancla pueden traer la misma relación repetida; por ejemplo, si Jon Snow y Daenerys comparten un vecino en su vecindario a 2 saltos, esa relación compartida se recuperaría una vez desde cada ancla. Quitamos esos duplicados exactos antes de construir el contexto final, para no inflar el prompt con información redundante."""
    seen = set()
    unique = []
    for triple in triples:
        key = (triple["subject"], triple["relation"], triple["object"])
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    return unique


def format_context(triples: list[dict]) -> str:
    """Convierte la lista de tripletas en texto legible en lenguaje natural, para incluirlo como contexto en el prompt final del LLM generador. No usamos el JSON crudo aquí porque un LLM redacta mejores respuestas cuando el contexto ya viene en forma de frases simples, en vez de tener que interpretar una estructura de datos anidada."""
    lines = []
    for triple in triples:
        line = f"- {triple['subject']} {triple['relation']} {triple['object']}"
        # El atributo "momento" es opcional; solo lo añadimos a la línea si realmente tiene contenido, para no ensuciar el contexto con paréntesis vacíos.
        if triple.get("moment"):
            line += f" ({triple['moment']})"
        lines.append(line)
    return "\n".join(lines)


# Este es el segundo LLM del pipeline, el que redacta la respuesta final. Nótese la instrucción explícita de no inventar información fuera del contexto y de admitir cuando el contexto no alcance: preferimos una respuesta honesta de "no tengo suficiente información" antes que una respuesta con apariencia de estar fundamentada en el grafo pero que en realidad esté rellenando huecos por su cuenta.
ANSWER_SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre Juego de Tronos usando ÚNICAMENTE la información del siguiente contexto, extraído de un grafo de conocimiento. No inventes información que no esté en el contexto. Si el contexto no es suficiente para responder, dilo explícitamente.

CONTEXTO (tripletas del grafo):
{context}
"""

ANSWER_USER_PROMPT = """Pregunta: {question}"""


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Reutilizamos la misma instancia de ChatDeepSeek para las dos llamadas del pipeline (extracción de entidades y generación de la respuesta final), ya que ambas comparten el mismo modelo y configuración; solo cambia la plantilla de prompt que se le encadena en cada caso.
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTION_SYSTEM_PROMPT),
        ("user", ENTITY_EXTRACTION_USER_PROMPT),
    ])
    extraction_chain = extraction_prompt | llm

    print(f"Pregunta: {QUESTION}\n")

    # Paso 1: identificar los nodos de partida a partir de la pregunta.
    anchor_entities = extract_anchor_entities(extraction_chain, QUESTION)
    print(f"Entidades ancla identificadas: {anchor_entities}")

    # Paso 2: recuperar el vecindario de esas entidades desde Neo4j.
    with driver.session() as session:
        triples = fetch_neighborhood(session, anchor_entities, MAX_HOPS)

    driver.close()

    unique_triples = deduplicate_triples(triples)
    print(f"Tripletas recuperadas del vecindario (tras deduplicar): {len(unique_triples)}")

    # Paso 3: convertir el subgrafo recuperado en texto legible.
    context = format_context(unique_triples)
    print(f"\n--- Contexto recuperado ---\n{context}\n")

    # Paso 4: generar la respuesta final apoyándose en ese contexto.
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        ("user", ANSWER_USER_PROMPT),
    ])
    answer_chain = answer_prompt | llm

    response = answer_chain.invoke({"context": context, "question": QUESTION})

    print("--- Respuesta final ---")
    print(response.content)


if __name__ == "__main__":
    main()
