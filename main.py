from imagelib.extractor import Extractor
import matplotlib.pyplot as plt

extractor = Extractor()
graph_list = extractor.extract("/Users/gwonchan-u/Downloads/Infant ABR 51-110/test11.png", 667, True)

print(len(graph_list[0]))

for graph_info in graph_list:
    graph = graph_info["graph"]
    peak_list = graph_info["peak"]
    plt.plot(range(len(graph)), graph, linewidth=1)
    for peak in peak_list:
        plt.scatter(peak[0], peak[1])

# print(graph_peak)
plt.legend()
plt.show()
