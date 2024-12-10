import requests
import os
import re
import json

base_url = "https://en.wikipedia.org/w/api.php"
output_dir = "wikipedia_pages"


# Fetch pages and subcategories of a given category.
def fetch_category_members(category):
    members = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "max",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(base_url, params=params)
        data = response.json()

        if "query" in data:
            members.extend(data["query"]["categorymembers"])

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return members


#Save a Wikipedia page's content to a file.
def save_page(page_data):
    filename =  sanitize_title(page_data["title"])
    filename = os.path.join(output_dir, f"{filename}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=4)  


def fetch_page_content(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if "missing" not in page_data:
            return  {
                "page_id": page_id,
                "title": page_data.get("title"),
                "url": "https://en.wikipedia.org/wiki/" + page_data.get("title"),
                "text": page_data.get("extract"),
            }
    return ""


# Recursively process a category
def process_category(category):
    members = fetch_category_members(category)
    for member in members:
        title = member["title"]
        if member["ns"] == 0:  # Namespace 0 indicates an article
            print(f"Downloading page: {title}")
            page_data = fetch_page_content(title)
            save_page(page_data)
        elif member["ns"] == 14:  # Namespace 14 indicates a category
            print(f"Descending into subcategory: {title}")
            process_category(title)
            
            
def sanitize_title(title):
    sanitized = re.sub(r'[\\\\/:*?"<>|]', '_', title)
    return sanitized



def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    category = "Category:Musical groups by genre"#"Category:English alternative rock groups" #"Category:Musical groups by genre"
    process_category(category)


if __name__ == "__main__":
    main()

