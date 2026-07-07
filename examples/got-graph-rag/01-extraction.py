"""
Este script lee el corpus, lo trocea por párrafos, y para cada fragmento le pide a DeepSeek que extraiga entidades y relaciones respetando estrictamente la ontología definida en ontology.json. Cada relación que devuelve el LLM se valida contra el dominio/rango declarado antes de aceptarla; lo que no encaja se descarta pero se guarda aparte, para poder revisarlo.

Lo que este script NO hace (a propósito, se deja para la Fase 3):
- No fusiona entidades duplicadas (ej. "Casa Stark" aparece varias veces).
- No valida las propiedades de las entidades, solo las relaciones.
Ambas cosas se resuelven al cargar los datos en Neo4j.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# Carga las variables definidas en el archivo .env (entre ellas, DEEPSEEK_API_KEY) como si fueran variables de entorno normales del sistema operativo. ChatDeepSeek las lee automáticamente, no hace falta pasarlas a mano.
load_dotenv()

# Rutas de los archivos que el script lee y escribe. Se asume que el script se ejecuta desde la carpeta del proyecto, donde ya deberían existir ontology.json y corpus.txt.
ONTOLOGY_PATH = Path("ontology.json")
CORPUS_PATH = Path("corpus.txt")
OUTPUT_PATH = Path("extraction.json")


def load_ontology() -> dict:
    """Carga el archivo ontology.json y lo devuelve como diccionario Python."""
    with open(ONTOLOGY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_chunks() -> list[str]:
    """
    Trocea el corpus en fragmentos (chunking).

    Decisión de diseño: usamos "un párrafo = un chunk", partiendo el texto por saltos de línea dobles (\n\n). No usamos ninguna librería de text-splitting (como las que trae LangChain) porque el corpus ya está naturalmente segmentado en unidades temáticas razonables, y añadir una librería más aquí sería complejidad innecesaria para este tamaño de texto.
    """
    text = CORPUS_PATH.read_text(encoding="utf-8")
    # Separamos por párrafo, quitamos espacios sobrantes y descartamos posibles fragmentos vacíos que puedan colarse al hacer split.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def build_ontology_description(ontology: dict) -> str:
    """
    Convierte el JSON de la ontología en un bloque de texto legible, pensado para ser leído tanto por un humano como por el LLM dentro del prompt del sistema.

    Por qué no le pasamos el JSON crudo al LLM: un LLM entiende mejor instrucciones en lenguaje natural con estructura tipo lista que un JSON anidado, y además así controlamos exactamente qué información de la ontología se expone (por ejemplo, aquí no exponemos el bloque "atributos_de_relacion", que se maneja aparte en las reglas del prompt).
    """
    lines = ["CLASES PERMITIDAS:"]
    for class_name, data in ontology["clases"].items():
        # Listamos los nombres de las propiedades esperadas para esa clase, para que el LLM sepa qué campos rellenar en "propiedades".
        props = ", ".join(data["propiedades"].keys())
        lines.append(f"- {class_name}: {data['comment']} (propiedades: {props})")

    lines.append("\nRELACIONES PERMITIDAS:")
    for rel in ontology["relaciones"]:
        # El dominio/rango puede ser un string ("Personaje") o una lista (["CasaNoble", "Organizacion"]) según cómo se definió en la ontología. Normalizamos ambos casos al mismo formato de texto "A | B" para que quede legible en el prompt.
        domain = rel["dominio"] if isinstance(rel["dominio"], str) else " | ".join(rel["dominio"])
        range_ = rel["rango"] if isinstance(rel["rango"], str) else " | ".join(rel["rango"])
        lines.append(f"- {rel['nombre']}: [{domain}] -> [{range_}] — {rel['comment']}")

    return "\n".join(lines)


# --- Plantillas de prompt ---
# SYSTEM_PROMPT recibe un placeholder {ontology_description} que se
# rellena en tiempo de ejecución con el resultado de build_ontology_description().
# Las llaves dobles {{ }} son necesarias porque este texto se usa como
# plantilla de LangChain (que usa llaves simples para variables), y el
# JSON de ejemplo que mostramos dentro también usa llaves — hay que
# "escaparlas" duplicándolas para que LangChain no las confunda con variables.
SYSTEM_PROMPT = """Eres un sistema de extracción de información para construir un grafo de conocimiento.

Debes identificar ÚNICAMENTE entidades y relaciones que encajen EXACTAMENTE en la siguiente ontología. Si una relación o entidad del texto no encaja en ninguna clase o relación permitida, NO la incluyas.

{ontology_description}

REGLAS ESTRICTAS:
1. Usa EXCLUSIVAMENTE los nombres de clase y relación tal como aparecen arriba (mismo formato, mayúsculas incluidas en las relaciones).
2. Respeta el dominio y rango de cada relación: el tipo del sujeto debe coincidir con el dominio, el tipo del objeto con el rango.
3. Si detectas que una relación ocurrió en un momento distinguible de la narrativa (por ejemplo, antes o después de un evento clave), añade el campo "momento" con una frase corta describiéndolo. Si no es relevante, omite el campo.
4. Responde ÚNICAMENTE con un JSON válido, sin texto adicional, sin explicaciones, sin bloques de código markdown.

FORMATO DE SALIDA (JSON estricto):
{{
  "entidades": [
    {{"nombre": "...", "tipo": "...", "propiedades": {{}}}}
  ],
  "relaciones": [
    {{"sujeto": "...", "tipo_sujeto": "...", "relacion": "...", "objeto": "...", "tipo_objeto": "...", "momento": "..."}}
  ]
}}
"""

# El prompt de usuario es más simple: solo inserta el fragmento de texto concreto que queremos procesar en esta llamada.
USER_PROMPT = """Extrae entidades y relaciones del siguiente fragmento de texto:

---
{fragment}
---

Responde solo con el JSON."""


def clean_json_response(text: str) -> str:
    """
    Limpia la respuesta bruta del LLM antes de intentar parsearla como JSON.

    Hace falta ya que, aunque en el prompt le pedimos explícitamente que NO use bloques de código markdown, en la práctica algunos modelos (DeepSeek incluido, a veces) devuelven la respuesta envuelta en ```json ... ``` de todos modos. Esta función detecta y quita esa envoltura si aparece, para que json.loads() no falle.
    """
    text = text.strip()
    if text.startswith("```"):
        # Nos quedamos con el contenido entre el primer y segundo ```
        text = text.split("```")[1]
        # Si el bloque empezaba con ```json, quitamos también la palabra "json"
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def validate_relation(rel: dict, ontology: dict) -> tuple[bool, str]:
    """
    Valida una relación extraída por el LLM contra la ontología.

    Esta función es el corazón de la fase de validación: no confiamos ciegamente en que el LLM haya respetado las reglas solo porque se lo pedimos en el prompt. Aquí comprobamos "en código" que:
      1. La relación realmente exista en la ontología (que no se haya inventado un nombre nuevo).
      2. El tipo del sujeto sea compatible con el dominio declarado.
      3. El tipo del objeto sea compatible con el rango declarado.

    Devuelve una tupla (es_valida, motivo). Si es_valida es False, motivo explica por qué, para poder guardarlo junto a la relación descartada y revisarlo después.
    """
    # Buscamos la definición de esta relación dentro de la lista de relaciones permitidas por su nombre exacto (ej. "ALIADA_DE").
    definition = next((r for r in ontology["relaciones"] if r["nombre"] == rel.get("relacion")), None)
    if definition is None:
        # El LLM devolvió un nombre de relación que no existe en la ontología.
        return False, f"Relación desconocida: {rel.get('relacion')}"

    # El dominio/rango en la ontología puede ser un string suelto ("Personaje") o una lista (["CasaNoble", "Organizacion"]). Normalizamos siempre a lista para poder usar "in" de forma uniforme.
    domain = definition["dominio"]
    domain = [domain] if isinstance(domain, str) else domain
    range_ = definition["rango"]
    range_ = [range_] if isinstance(range_, str) else range_

    if rel.get("tipo_sujeto") not in domain:
        return False, f"'{rel.get('relacion')}' no admite sujeto de tipo '{rel.get('tipo_sujeto')}' (dominio válido: {domain})"
    if rel.get("tipo_objeto") not in range_:
        return False, f"'{rel.get('relacion')}' no admite objeto de tipo '{rel.get('tipo_objeto')}' (rango válido: {range_})"

    # Si llegamos aquí, la relación existe y respeta dominio y rango.
    return True, ""


def main():
    # --- Paso 1: cargar los insumos ---
    ontology = load_ontology()
    chunks = load_chunks()
    ontology_description = build_ontology_description(ontology)

    # --- Paso 2: configurar el modelo ---
    # temperature=0 porque queremos extracción determinista y consistente, no creatividad. Para tareas de extracción estructurada, la temperatura alta solo añade variabilidad indeseada entre ejecuciones.
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    # ChatPromptTemplate.from_messages construye la plantilla de conversación que le enviaremos al modelo: un mensaje de sistema (con las reglas y la ontología) y un mensaje de usuario (con el fragmento a analizar).
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    # El operador | encadena la plantilla de prompt con el modelo: esto es la sintaxis de LCEL (LangChain Expression Language). Al invocar "chain", primero se rellena el prompt con las variables y después el resultado se envía automáticamente al LLM.
    chain = prompt | llm

    # Listas donde iremos acumulando los resultados de TODOS los párrafos, no solo del último procesado.
    all_entities = []
    all_relations = []
    discarded_relations = []

    # --- Paso 3: procesar cada párrafo (chunk) por separado ---
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Procesando párrafo {i}/{len(chunks)} ---")

        # Invocamos la cadena: esto rellena el prompt con la descripción de la ontología y el fragmento actual, y hace la llamada real a la API de DeepSeek. response.content contiene el texto devuelto por el modelo.
        response = chain.invoke({
            "ontology_description": ontology_description,
            "fragment": chunk,
        })

        # Intentamos interpretar la respuesta como JSON. Si el modelo devolvió algo que no es JSON válido (puede pasar, los LLMs no son perfectos), lo registramos y pasamos al siguiente párrafo en vez de detener todo el script.
        try:
            data = json.loads(clean_json_response(response.content))
        except json.JSONDecodeError:
            print("  Respuesta no es JSON válido, se omite este párrafo.")
            continue

        # Si alguna clave falta en la respuesta, usamos lista vacía por defecto.
        entities = data.get("entidades", [])
        relations = data.get("relaciones", [])

        # Las entidades NO se validan en este script (ver docstring inicial), así que se añaden directamente al resultado global.
        all_entities.extend(entities)
        print(f"  Entidades detectadas: {len(entities)}")

        # Cada relación SÍ pasa por validate_relation antes de aceptarse.
        valid_count = 0
        for rel in relations:
            is_valid, reason = validate_relation(rel, ontology)
            if is_valid:
                all_relations.append(rel)
                valid_count += 1
            else:
                # Guardamos la relación descartada junto con el motivo, en vez de simplemente ignorarla. Esto es clave para poder auditar después qué intentó hacer el LLM y por qué se rechazó (por ejemplo, para detectar fallos de diseño en la propia ontología, como nos pasó con LEAL_A y GOBIERNA).
                discarded_relations.append({**rel, "motivo_descarte": reason})
                print(f" Relación descartada: {reason}")

        print(f"  Relaciones válidas: {valid_count}/{len(relations)}")

    # --- Paso 4: guardar el resultado consolidado ---
    result = {
        "entidades": all_entities,
        "relaciones": all_relations,
        "relaciones_descartadas": discarded_relations,
    }

    # ensure_ascii=False para que los acentos y "ñ" se guarden legibles en el archivo, en vez de como secuencias de escape tipo ñ. indent=2 para que el JSON resultante sea fácil de leer a simple vista.
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RESUMEN ===")
    print(f"Entidades extraídas: {len(all_entities)}")
    print(f"Relaciones válidas: {len(all_relations)}")
    print(f"Relaciones descartadas: {len(discarded_relations)}")
    print(f"Guardado en: {OUTPUT_PATH}")


# Este bloque solo se ejecuta si corremos el archivo directamente (python extraction.py o uv run extraction.py), no si se importa como módulo desde otro script.
if __name__ == "__main__":
    main()