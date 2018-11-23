import numpy as np

v = 10
graph = np.random.randint(0, 2, (v, v))

row = np.insert(np.cumsum(np.sum(graph, axis=1)), 0, 0)
col = np.where(graph == 1)[1]

with open("data.txt","w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in ([v, np.sum(graph)], row, col)))
