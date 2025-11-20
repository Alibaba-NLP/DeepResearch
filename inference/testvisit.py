from tool_visitLOCAL import Visit
from db_min import fetch_document, reset_docs
import re

reset_docs()

tool = Visit()
obs = tool.call({
    "url": "https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration",
    "goal": "Give me the best cars",
    "question_id": 25
})

print("OBS 1st line:", obs.splitlines()[0])
m = re.search(r"\[\[DOC_ID:(\d+)\]\]", obs)
assert m, "No [[DOC_ID:...]] tag found in tool observation!"
doc_id = int(m.group(1))
print("Captured doc_id:", doc_id)

content = fetch_document(doc_id)
print("DB content preview:", (content or "")[:200])
