"""
Este script trae el grafo completo desde Neo4j, lo reconstruye en memoria con NetworkX, detecta comunidades (clusters de nodos densamente conectados) con el algoritmo de Louvain, guarda de vuelta en Neo4j a qué comunidad pertenece cada nodo, y genera con DeepSeek un resumen en lenguaje natural de cada comunidad detectada.

Por qué NetworkX y no el plugin Graph Data Science de Neo4j: como vimos en el plan inicial, AuraDB Free no incluye GDS (solo disponible en planes de pago), así que traemos el grafo a Python y usamos la implementación de Louvain que ya trae NetworkX de serie, sin depender de infraestructura adicional.

Por qué esta fase importa para la Fase 6 (Global Search): una pregunta como "¿cuáles son las principales facciones en conflicto?" no parte de ningún nodo concreto del grafo, así que no se puede resolver con un recorrido tipo Local Search. La solución es resumir el grafo completo en bloques manejables por adelantado (aquí), para que en el momento de la pregunta el sistema lea esos resúmenes ya hechos en vez de tener que procesar el grafo entero de golpe.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

load_dotenv()

OUTPUT_PATH = Path("communities.json")

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


def fetch_graph(session) -> tuple[list[dict], list[dict]]:
    """Trae todos los nodos y relaciones de Neo4j como listas de diccionarios simples de Python, en vez de como los objetos propios del driver (Node, Relationship), para poder pasárselos directamente a NetworkX sin conversiones intermedias."""
    # normalizedName es el mismo identificador que usamos como clave de fusión en la Fase 3, así que lo reutilizamos aquí como identificador de nodo dentro de NetworkX: así garantizamos que "el mismo nodo" en Neo4j y en el grafo de NetworkX comparten exactamente la misma clave.
    nodes_result = session.run("""
        MATCH (n)
        RETURN n.normalizedName AS id, n.nombre AS name, labels(n) AS labels
    """)
    nodes = [dict(record) for record in nodes_result]

    # Aquí solo recuperamos el tipo de relación (type(r)), no sus propiedades como "momento", porque para la detección de comunidades lo único que le importa a Louvain es que existe una conexión entre dos nodos, no los detalles de esa conexión.
    edges_result = session.run("""
        MATCH (s)-[r]->(o)
        RETURN s.normalizedName AS source, o.normalizedName AS target, type(r) AS relation_type
    """)
    edges = [dict(record) for record in edges_result]

    return nodes, edges


def build_networkx_graph(nodes: list[dict], edges: list[dict]) -> nx.Graph:
    """Construye un grafo NO dirigido de NetworkX a partir de los nodos y relaciones. Usamos un grafo no dirigido a propósito: Louvain detecta comunidades por densidad de conexiones, sin importar el sentido de la flecha, así que para efectos de "quién está en el mismo bando" da igual que el dato original fuera Stark ALIADA_DE Tully o Tully ALIADA_DE Stark, es la misma señal estructural de que ambos nodos están conectados."""
    graph = nx.Graph()
    # Añadimos primero todos los nodos, aunque algunos puedan quedar sin ninguna relación (nodos aislados); NetworkX los mantiene igualmente en el grafo, aunque Louvain los tratará casi con seguridad como su propia comunidad de un solo miembro.
    for node in nodes:
        graph.add_node(node["id"], name=node["name"], labels=node["labels"])
    # add_edge crea automáticamente los nodos si no existieran ya (no debería pasar aquí, porque ya los añadimos arriba), y si la misma arista se añade dos veces, NetworkX simplemente actualiza sus propiedades en vez de duplicarla, ya que un grafo simple de NetworkX no permite aristas paralelas entre el mismo par de nodos.
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], relation_type=edge["relation_type"])
    return graph


def detect_communities(graph: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Ejecuta el algoritmo de Louvain sobre el grafo y devuelve un diccionario {id_de_nodo: id_de_comunidad}, que es la forma más práctica de consultar después a qué comunidad pertenece cualquier nodo concreto."""
    # louvain_communities devuelve una lista de conjuntos (sets), donde cada conjunto contiene los identificadores de los nodos que Louvain agrupó como una misma comunidad; el número de comunidades no se decide de antemano, lo determina el propio algoritmo optimizando la modularidad del grafo.
    # resolution controla el tamaño típico de las comunidades: valores por debajo de 1 favorecen comunidades más grandes y menos numerosas, valores por encima de 1 favorecen más comunidades, más pequeñas. En grafos pequeños como el nuestro, el valor por defecto (1.0) tiende a fragmentar demasiado, dejando nodos de bajo grado (como las regiones, que solo tienen una relación GOBIERNA) como comunidades de un solo miembro; bajar la resolution empuja al algoritmo a fusionar esos nodos con la comunidad de su vecino más cercano en vez de aislarlos.
    # seed=42 fija la semilla del generador aleatorio interno del algoritmo: Louvain evalúa los nodos en un orden parcialmente aleatorio, así que sin fijar la semilla, dos ejecuciones sobre el mismo grafo podrían dar particiones ligeramente distintas. Fijarla nos da reproducibilidad, algo valioso para un tutorial donde queremos poder comparar resultados de forma consistente.
    communities = nx.algorithms.community.louvain_communities(graph, resolution=resolution, seed=42)
    node_to_community = {}
    # enumerate nos da un índice numérico (0, 1, 2...) para cada comunidad detectada, que usamos como su identificador.
    for community_id, members in enumerate(communities):
        for node_id in members:
            node_to_community[node_id] = community_id
    return node_to_community


def save_communities_to_neo4j(session, node_to_community: dict[str, int]):
    """Escribe de vuelta en Neo4j la propiedad communityId de cada nodo, para que el resultado de la detección de comunidades quede persistido en la base de datos y disponible para futuras consultas Cypher, sin tener que volver a ejecutar Louvain cada vez que alguien quiera saber a qué comunidad pertenece un nodo."""
    for node_id, community_id in node_to_community.items():
        session.run(
            "MATCH (n {normalizedName: $node_id}) SET n.communityId = $community_id",
            node_id=node_id,
            community_id=community_id,
        )


def group_members_by_community(nodes: list[dict], node_to_community: dict[str, int]) -> dict[int, list[dict]]:
    """Reorganiza la lista plana de nodos en un diccionario {id_de_comunidad: [nodos]}, que es la forma que necesitamos para generar un resumen por comunidad en el siguiente paso: para escribir el resumen de la comunidad 0, necesitamos tener ya agrupados todos sus miembros juntos, no dispersos en una lista plana."""
    grouped = defaultdict(list)
    for node in nodes:
        community_id = node_to_community.get(node["id"])
        # Un nodo podría quedar sin comunidad asignada solo si estuviera completamente desconectado del resto del grafo y Louvain no lo hubiera procesado; esta comprobación es una salvaguarda defensiva para ese caso extremo.
        if community_id is not None:
            grouped[community_id].append(node)
    return grouped


# Este prompt le pide al LLM que actúe como analista, no como extractor de datos: a diferencia de la Fase 2 (donde queríamos precisión estructural estricta), aquí el objetivo es una síntesis interpretativa en lenguaje natural, así que el tono de las instrucciones es deliberadamente distinto.
SUMMARY_SYSTEM_PROMPT = """Eres un analista que resume comunidades detectadas en un grafo de conocimiento sobre Juego de Tronos. Se te dará una lista de entidades (personajes, casas, organizaciones, etc.) que el algoritmo de detección de comunidades agrupó por estar densamente conectadas entre sí. Tu trabajo es escribir un resumen breve (2-4 frases) en español que explique qué representa este grupo: por ejemplo, si son una facción, una alianza, un bando enfrentado a otro, etc. Basa el resumen únicamente en los nombres de las entidades que se te dan, sin inventar información que no puedas inferir razonablemente de esos nombres."""

SUMMARY_USER_PROMPT = """Entidades de esta comunidad:
{members}

Escribe el resumen."""


def generate_community_summary(llm, chain, members: list[dict]) -> str:
    """Llama a DeepSeek para generar el resumen en lenguaje natural de una comunidad concreta, a partir de la lista de nombres de sus miembros y su tipo (Personaje, CasaNoble...), que le da al LLM una pista adicional sobre la naturaleza de cada entidad sin tener que consultar sus propiedades completas."""
    # labels(n) en Neo4j devuelve una lista (por si un nodo tuviera varias etiquetas a la vez), así que tomamos el primer elemento, ya que en nuestra ontología cada nodo tiene exactamente un tipo.
    member_names = ", ".join(f"{m['name']} ({m['labels'][0]})" for m in members)
    response = chain.invoke({"members": member_names})
    return response.content.strip()


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Todo el trabajo que necesita la base de datos (leer el grafo y luego escribir communityId) se hace dentro de una única sesión, para reutilizar la misma conexión en vez de abrir y cerrar sesiones innecesariamente.
    with driver.session() as session:
        nodes, edges = fetch_graph(session)
        print(f"Nodos traídos de Neo4j: {len(nodes)}")
        print(f"Relaciones traídas de Neo4j: {len(edges)}")

        graph = build_networkx_graph(nodes, edges)
        node_to_community = detect_communities(graph, resolution=0.5)

        num_communities = len(set(node_to_community.values()))
        print(f"Comunidades detectadas: {num_communities}")

        save_communities_to_neo4j(session, node_to_community)

    driver.close()

    # A partir de aquí ya no necesitamos la conexión a Neo4j: el resto del script trabaja solo con los datos que ya trajimos a memoria (nodes, node_to_community) y con llamadas al LLM.
    grouped = group_members_by_community(nodes, node_to_community)

    # temperature=0.3 es una diferencia deliberada respecto a la Fase 2 (donde usamos 0 para extracción determinista): aquí queremos que el resumen suene natural y bien redactado, no una transcripción mecánica, así que permitimos un poco más de variabilidad sin llegar a los niveles altos que fomentarían que el modelo invente información no presente en los nombres.
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("user", SUMMARY_USER_PROMPT),
    ])
    chain = prompt | llm

    communities_output = []
    # sorted(grouped.items()) simplemente nos asegura que los resúmenes se impriman y se guarden en orden de id de comunidad (0, 1, 2...), en vez de en un orden arbitrario dependiente de cómo Python recorra el diccionario internamente.
    for community_id, members in sorted(grouped.items()):
        print(f"\n--- Generando resumen de la comunidad {community_id} ({len(members)} miembros) ---")
        summary = generate_community_summary(llm, chain, members)
        print(f"  {summary}")
        communities_output.append({
            "community_id": community_id,
            "members": [m["name"] for m in members],
            "summary": summary,
        })

    # Este archivo, communities.json, es exactamente lo que la Fase 6 (Global Search) va a leer en vez de tener que volver a consultar Neo4j o recalcular las comunidades desde cero.
    OUTPUT_PATH.write_text(json.dumps(communities_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()