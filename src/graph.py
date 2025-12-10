from neo4j import GraphDatabase

class Graph:

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="admin123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))


    def close(self):
        self.driver.close()

    def get_entity_subgraph(self, entity_name):

        query = """
        MATCH (n {name:$name})

        // Artisti che influenzano l'entità
        OPTIONAL MATCH (influencer)<-[:influenced_by]-(n)

        // Artisti influenzati dall'entità
        OPTIONAL MATCH (influenced)-[:influenced_by]->(n)

        RETURN 
            [x IN collect(DISTINCT influencer) | x.name + " (" + coalesce(x.nationality,'Unknown') + ")"] AS influenced_by,
            [x IN collect(DISTINCT influenced) | x.name + " (" + coalesce(x.nationality,'Unknown') + ")"] AS influences
        """
        with self.driver.session() as session:
            record = session.run(query, name=entity_name).single()
            if not record:
                return {"influenced_by": [], "influences": []}
            return record.data()


    def verbalize_rag_context(self, entity_name, subgraph_data):
        lines = []

        def format_list(items):
            formatted = []
            for item in items:
                if "(" in item:
                    name, nat = item.rsplit("(", 1)
                    nat = nat.rstrip(")")
                    if nat.lower() != "unknown":
                        formatted.append(f"    - {name.strip()} ({nat})")
                    else:
                        formatted.append(f"    - {name.strip()}")
                else:
                    formatted.append(f"    - {item.strip()}")
            return formatted

        if subgraph_data.get("influenced_by"):
            lines.append(entity_name + " is influenced by:")
            lines.extend(format_list(subgraph_data["influenced_by"]))

        if subgraph_data.get("influences"):
            lines.append(entity_name + " has influenced:")
            lines.extend(format_list(subgraph_data["influences"]))

        lines.append("\nNote: nationality is shown only when available.")

        return "\n".join(lines)


