"""
Este script implementa el patrón map-reduce sobre los resúmenes de comunidades generados en la Fase 4, evaluando la relevancia de cada comunidad frente a la pregunta y sintetizando las más relevantes en una única respuesta final.

Por qué esto no se resuelve con un recorrido de grafo (como en Local Search): la pregunta "¿cuáles son las principales facciones en conflicto?" no menciona ningún nodo concreto del grafo, así que no hay ningún punto de partida desde el cual hacer MATCH en Cypher. En vez de eso, el sistema se apoya en el trabajo ya hecho en la Fase 4: los resúmenes de cada comunidad, generados de antemano, que resumen de qué trata cada bloque densamente conectado del grafo.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

COMMUNITIES_PATH = Path("communities.json")

QUESTION = "¿Cuáles son las principales facciones en conflicto y qué las mueve?"

# Solo se envían a la fase "reduce" las comunidades cuya relevancia (puntuada de 0 a 10 en la fase "map") sea igual o superior a este umbral. Esto es lo que debería filtrar de forma natural las comunidades de una sola región geográfica que vimos en la Fase 4, sin tener que limpiarlas a mano del archivo communities.json.
RELEVANCE_THRESHOLD = 5


def load_communities() -> list[dict]:
    with open(COMMUNITIES_PATH, encoding="utf-8") as f:
        return json.load(f)


# Este prompt es la fase "map": se ejecuta una vez POR CADA comunidad, de forma independiente, sin que una llamada sepa nada de las demás comunidades. Le pedimos al LLM que puntúe la relevancia y que extraiga solo la información pertinente a la pregunta, no que redacte ya una respuesta final, ya que eso corresponde a la fase "reduce".
MAP_SYSTEM_PROMPT = """Eres un analista evaluando si el resumen de una comunidad de un grafo de conocimiento es relevante para responder una pregunta concreta. Se te dará la pregunta y el resumen de una comunidad. Debes devolver ÚNICAMENTE un JSON con esta forma: {{"relevancia": <número entero de 0 a 10>, "extracto": "<una o dos frases con la información de esta comunidad que sea útil para responder la pregunta, o cadena vacía si la relevancia es baja>"}}. No incluyas texto adicional fuera del JSON."""

MAP_USER_PROMPT = """Pregunta: {question}

Resumen de la comunidad:
{summary}

Miembros de la comunidad: {members}

Responde solo con el JSON."""


def map_community(chain, question: str, community: dict) -> dict:
    """Evalúa una única comunidad frente a la pregunta, devolviendo su relevancia y un extracto de información útil. Esta es la unidad de trabajo que se repite una vez por comunidad en la fase "map"."""
    response = chain.invoke({
        "question": question,
        "summary": community["summary"],
        "members": ", ".join(community["members"]),
    })
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    return {
        "community_id": community["community_id"],
        "relevance": data.get("relevancia", 0),
        "extract": data.get("extracto", ""),
    }


# Este prompt es la fase "reduce": se ejecuta UNA SOLA VEZ, con los extractos ya filtrados de las comunidades relevantes, y su trabajo es sintetizar todo eso en una respuesta final coherente y bien redactada, en vez de simplemente concatenar los extractos sueltos.
REDUCE_SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre Juego de Tronos a partir de varios extractos de información, cada uno proveniente de una comunidad distinta de un grafo de conocimiento. Sintetiza estos extractos en una única respuesta coherente y bien redactada en español, organizada por facción cuando tenga sentido. No inventes información que no esté en los extractos proporcionados."""

REDUCE_USER_PROMPT = """Pregunta: {question}

Extractos relevantes recopilados:
{extracts}

Redacta la respuesta final."""


def reduce_extracts(chain, question: str, mapped_communities: list[dict]) -> str:
    """Combina los extractos de las comunidades relevantes en una única respuesta final, delegando la síntesis en el LLM en vez de simplemente concatenar texto."""
    extracts_text = "\n".join(
        f"- (Comunidad {c['community_id']}, relevancia {c['relevance']}/10): {c['extract']}"
        for c in mapped_communities
    )
    response = chain.invoke({"question": question, "extracts": extracts_text})
    return response.content


def main():
    communities = load_communities()
    print(f"Comunidades cargadas: {len(communities)}")
    print(f"Pregunta: {QUESTION}\n")

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", MAP_SYSTEM_PROMPT),
        ("user", MAP_USER_PROMPT),
    ])
    map_chain = map_prompt | llm

    # Fase "map": evaluamos cada comunidad de forma independiente. Aquí lo hacemos de forma secuencial (un bucle simple) para mantener el script fácil de seguir; en un sistema en producción, estas llamadas serían buenas candidatas a paralelizarse, ya que ninguna depende del resultado de las demás.
    mapped_communities = []
    for community in communities:
        result = map_community(map_chain, QUESTION, community)
        mapped_communities.append(result)
        print(f"Comunidad {result['community_id']}: relevancia {result['relevance']}/10")

    # Filtramos por el umbral de relevancia antes de pasar a la fase "reduce", y ordenamos de mayor a menor relevancia para que la comunidad más pertinente aparezca primero en el contexto que le demos al LLM final.
    relevant_communities = [c for c in mapped_communities if c["relevance"] >= RELEVANCE_THRESHOLD]
    relevant_communities.sort(key=lambda c: c["relevance"], reverse=True)

    print(f"\nComunidades relevantes (relevancia >= {RELEVANCE_THRESHOLD}): {len(relevant_communities)} de {len(communities)}")

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", REDUCE_SYSTEM_PROMPT),
        ("user", REDUCE_USER_PROMPT),
    ])
    reduce_chain = reduce_prompt | llm

    # Fase "reduce": una única llamada final que sintetiza los extractos ya filtrados en una respuesta coherente.
    answer = reduce_extracts(reduce_chain, QUESTION, relevant_communities)

    print("\n--- Respuesta final ---")
    print(answer)


if __name__ == "__main__":
    main()
