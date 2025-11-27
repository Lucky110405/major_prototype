import requests
import sys
import json

BASE = "http://localhost:6333"

def collection_info(name: str):
    url = f"{BASE}/collections/{name}"
    r = requests.get(url)
    try:
        r.raise_for_status()
    except Exception as e:
        print(f"Error fetching collection info: {e}")
        print(r.text)
        return None
    return r.json()

def search_collection(name: str, vector, top_k: int = 5):
    url = f"{BASE}/collections/{name}/points/search"
    body = {"vector": vector, "limit": top_k, "with_payload": True}
    r = requests.post(url, json=body)
    try:
        r.raise_for_status()
    except Exception as e:
        print(f"Error searching collection: {e}")
        print(r.text)
        return None
    return r.json()

def main():
    if len(sys.argv) < 2:
        print("Usage: python qdrant_diag.py <collection> [query_vector_json] [top_k]")
        print("If query_vector_json is omitted, a random probe will be used (not semantic).")
        return

    collection = sys.argv[1]
    info = collection_info(collection)
    print(json.dumps(info, indent=2))

    if len(sys.argv) >= 3:
        try:
            vector = json.loads(sys.argv[2])
        except Exception:
            print("Failed to parse vector JSON. Provide a JSON array of floats.")
            return
    else:
        # simple probe vector of zeros (only useful to test the API)
        vector = [0.0] * (info.get("result", {}).get("config", {}).get("params", {}).get("vectors", {}).get("size", 768))

    top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    res = search_collection(collection, vector, top_k=top_k)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
