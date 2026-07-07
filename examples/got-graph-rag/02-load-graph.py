"""
Este script lee extraction.json y ontology.json, resuelve entidades duplicadas (fusionando por nombre normalizado), y carga nodos y relaciones en Neo4j usando MERGE, expandiendo automáticamente las relaciones simétricas en ambos sentidos.

Resolución de entidades en este script (a propósito, nivel básico): normalizamos el nombre (minúsculas, sin tildes, sin espacios extra) y usamos ese valor normalizado como clave de fusión en Cypher. Esto resuelve duplicados exactos ("Casa Stark" repetido) y casi-duplicados triviales ("daenerys" vs "Daenerys"), pero NO resuelve alias distintos como "Jon" vs "Jon Snow" - eso requeriría fuzzy matching o un paso adicional con el LLM, que dejamos fuera para no complicar el tutorial.
"""

import json
import os
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

ONTOLOGY_PATH = Path("ontology.json")
EXTRACTION_PATH = Path("extraction.json")

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    """Normaliza un nombre para usarlo como clave de fusión de entidades. Pasamos a minúsculas y quitamos tildes (vía unicodedata), para que "Daenerys" y "daenerys" se traten como la misma entidad. No quitamos espacios internos porque "Jon Snow" y "Jon" deben seguir siendo nombres distintos en esta versión simple del script."""
    text = name.strip().lower()
    # Descompone caracteres acentuados (é -> e + acento) y luego nos quedamos solo con los caracteres que no son marcas diacríticas, lo que efectivamente elimina las tildes sin tocar el resto del texto.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def build_symmetric_relation_names(ontology: dict) -> set[str]:
    """Devuelve el conjunto de nombres de relación marcados como simétricos en la ontología."""
    return {rel["nombre"] for rel in ontology["relaciones"] if rel.get("simetrica")}


def load_entities(tx, entities: list[dict]):
    """Carga (o fusiona) cada entidad como nodo en Neo4j. Usamos MERGE sobre la propiedad normalizedName, que es la clave de resolución de entidades. La etiqueta del nodo (Personaje, CasaNoble...) se inserta de forma dinámica en la consulta Cypher, ya que Cypher no permite parametrizar labels directamente. SET n += $properties fusiona (no sobreescribe con null) las propiedades nuevas sobre las que ya tuviera el nodo, así que si una aparición trae "sede" y otra no, el dato no se pierde entre fusiones sucesivas."""
    for entity in entities:
        node_type = entity.get("tipo")
        name = entity.get("nombre")
        if not node_type or not name:
            continue
        properties = entity.get("propiedades", {}) or {}
        # Guardamos también el nombre original (con mayúsculas/tildes) como propiedad "nombre", ya que normalizedName solo se usa internamente para la fusión, no para mostrarlo al usuario.
        properties["nombre"] = name
        query = f"""
        MERGE (n:{node_type} {{normalizedName: $normalized_name}})
        SET n += $properties
        """
        tx.run(
            query,
            normalized_name=normalize_name(name),
            properties=properties,
        )


def load_relations(tx, relations: list[dict], symmetric_names: set[str]):
    """Carga cada relación como una arista en Neo4j. Para relaciones simétricas, creamos la arista en ambos sentidos (sujeto->objeto y objeto->sujeto), ya que la ontología declara que si una es cierta, la inversa también lo es, y preferimos guardarlo explícito en el grafo en vez de tener que recordarlo en cada consulta."""
    for rel in relations:
        relation_name = rel.get("relacion")
        subject_name = rel.get("sujeto")
        object_name = rel.get("objeto")
        if not relation_name or not subject_name or not object_name:
            continue
        # El atributo "momento" es opcional; si no viene, guardamos cadena vacía para mantener el esquema de propiedades uniforme.
        moment = rel.get("momento", "")
        # Cypher no permite parametrizar el tipo de relación, así que lo insertamos con f-string. Es seguro aquí porque relation_name viene siempre de la validación contra la ontología, no de texto libre arbitrario.
        query = f"""
        MATCH (s {{normalizedName: $subject_name}})
        MATCH (o {{normalizedName: $object_name}})
        MERGE (s)-[r:{relation_name}]->(o)
        SET r.momento = $moment
        """
        tx.run(
            query,
            subject_name=normalize_name(subject_name),
            object_name=normalize_name(object_name),
            moment=moment,
        )
        # Si la relación es simétrica, repetimos la operación invirtiendo sujeto y objeto, para dejar también la arista en sentido contrario.
        if relation_name in symmetric_names:
            tx.run(
                query,
                subject_name=normalize_name(object_name),
                object_name=normalize_name(subject_name),
                moment=moment,
            )


def main():
    ontology = load_json(ONTOLOGY_PATH)
    extraction = load_json(EXTRACTION_PATH)
    symmetric_names = build_symmetric_relation_names(ontology)
    entities = extraction.get("entidades", [])
    relations = extraction.get("relaciones", [])
    print(f"Entidades a cargar (antes de fusionar duplicados): {len(entities)}")
    print(f"Relaciones a cargar: {len(relations)}")
    print(f"Relaciones simétricas en la ontología: {sorted(symmetric_names)}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        # execute_write envuelve la operación en una transacción: si algo falla a mitad de la carga, Neo4j revierte los cambios de ese bloque en vez de dejar el grafo a medio cargar.
        session.execute_write(load_entities, entities)
        session.execute_write(load_relations, relations, symmetric_names)
        # Contamos cuántos nodos y relaciones distintos quedaron en la base de datos tras la fusión, para comparar contra los totales "en bruto" de arriba y ver cuánto se redujo por deduplicación.
        node_count = session.run("MATCH (n) RETURN count(n) AS total").single()["total"]
        relation_count = session.run("MATCH ()-[r]->() RETURN count(r) AS total").single()["total"]
    driver.close()
    print("\n=== RESUMEN ===")
    print(f"Nodos finales en Neo4j: {node_count}")
    print(f"Relaciones finales en Neo4j: {relation_count}")


if __name__ == "__main__":
    main()