def classify_query(query):
    query = query.lower()

    if "how many" in query or "number" in query:
        return "numeric"
    elif "who" in query:
        return "person"
    elif "where" in query:
        return "location"
    elif "what is" in query:
        return "description"
    else:
        return "other"
