
from tool_searchLOCAL import Search
t = Search()
print(t.call({"query": "test"}))
print("\n=======\n")
print(t.call({"query": ["deep learning","site:wikipedia.org reinforcement learning"]}))

