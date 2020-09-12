from imagelib.extractor import Extractor
import matplotlib.pyplot as plt

extractor = Extractor()
graph_list = extractor.extract_with_peak("/Users/gwonchan-u/Downloads/Infant ABR 51-110/test9.png", True)

# print(graph_list)

for graph_info in graph_list:
    graph = graph_info["graph"]
    peak_list = graph_info["peak"]
    plt.plot(range(len(graph)), graph, linewidth=1)
    for peak in peak_list:
        plt.scatter(peak[0], peak[1])

# print(graph_peak)
plt.legend()
plt.show()
