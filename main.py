import re
import sys
import random
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

#注释测试

class TextGraph:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.words = set()
        self.page_rank = {}

    def process_text(self, text):
        # 替换所有非字母字符为空格，并将换行符也替换为空格
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        # 转换为小写并分割单词
        words = text.lower().split()
        return words

    def build_graph(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        words = self.process_text(text)
        if not words:
            raise ValueError("文件不包含有效单词")

        self.words = set(words)

        # 构建有向图
        for i in range(len(words) - 1):
            from_word = words[i]
            to_word = words[i + 1]

            if to_word in self.graph[from_word]:
                self.graph[from_word][to_word] += 1
            else:
                self.graph[from_word][to_word] = 1

        # 初始化PageRank
        self._init_page_rank()

    def _init_page_rank(self, damping_factor=0.85, max_iter=100, tol=1e-6):
        num_nodes = len(self.words)
        if num_nodes == 0:
            return

        # 初始PR值为1/N
        pr = {word: 1.0 / num_nodes for word in self.words}

        for _ in range(max_iter):
            new_pr = {}
            # 计算所有出度为0的节点的PR值总和
            dangling_sum = sum(pr[word] for word in self.words if not self.graph[word])
            # 均分给所有节点（包括阻尼因子）
            dangling_contribution = damping_factor * dangling_sum / num_nodes

            # 计算每个节点的新PR值
            for word in self.words:
                # 来自其他节点的贡献
                incoming = 0.0
                for from_word in self.words:
                    if word in self.graph[from_word]:
                        outgoing_links = sum(self.graph[from_word].values())
                        incoming += pr[from_word] * self.graph[from_word][word] / outgoing_links

                # 随机跳转 + 常规贡献 + 漏出贡献
                new_pr[word] = (1 - damping_factor) / num_nodes + damping_factor * incoming + dangling_contribution

            # 检查收敛
            diff = sum(abs(new_pr[word] - pr[word]) for word in self.words)
            if diff < tol:
                break

            pr = new_pr

        self.page_rank = pr

    def show_directed_graph(self, save_path=None):
        # 使用networkx创建图形
        G = nx.DiGraph()

        # 添加节点和边
        for from_word in self.graph:
            for to_word, weight in self.graph[from_word].items():
                G.add_edge(from_word, to_word, weight=weight)

        # 绘制图形
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # 使用固定种子使布局一致
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue',
                font_size=10, font_weight='bold', arrowsize=20)

        # 添加边权重标签
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Directed Word Graph")

        if save_path:
            plt.savefig(save_path)
            print(f"图形已保存到 {save_path}")
        else:
            plt.show()

        # 在命令行中打印图形结构
        print("\n有向图结构：")
        for from_word in sorted(self.graph.keys()):
            for to_word, weight in sorted(self.graph[from_word].items()):
                print(f"{from_word} -> {to_word} (权重: {weight})")

    def query_bridge_words(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.words or word2 not in self.words:
            missing = []
            if word1 not in self.words:
                missing.append(f'"{word1}"')
            if word2 not in self.words:
                missing.append(f'"{word2}"')
            return f"No {' or '.join(missing)} in the graph!"

        bridge_words = []
        # 查找word1的后继节点
        successors = set(self.graph[word1].keys())
        # 查找word2的前驱节点
        predecessors = set()
        for word in self.graph:
            if word2 in self.graph[word]:
                predecessors.add(word)

        # 桥接词是同时是word1的后继和word2的前驱
        bridge_words = list(successors & predecessors)

        if not bridge_words:
            return f"No bridge words from \"{word1}\" to \"{word2}\"!"
        else:
            if len(bridge_words) == 1:
                return f"The bridge words from \"{word1}\" to \"{word2}\" is: \"{bridge_words[0]}\""
            else:
                bridge_str = ', '.join([f'"{w}"' for w in bridge_words[:-1]])
                bridge_str += f' and "{bridge_words[-1]}"'
                return f"The bridge words from \"{word1}\" to \"{word2}\" are: {bridge_str}."

    def generate_new_text(self, input_text):
        words = self.process_text(input_text)
        if len(words) < 2:
            return input_text

        new_words = [words[0]]

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            # 检查桥接词
            if word1 in self.words and word2 in self.words:
                successors = set(self.graph[word1].keys())
                predecessors = set()
                for word in self.graph:
                    if word2 in self.graph[word]:
                        predecessors.add(word)

                bridge_words = list(successors & predecessors)
                if bridge_words:
                    # 随机选择一个桥接词
                    bridge = random.choice(bridge_words)
                    new_words.append(bridge)

            new_words.append(word2)

        return ' '.join(new_words)

    def calc_shortest_path(self, word1, word2=None):
        word1 = word1.lower()
        if word1 not in self.words:
            return f"No \"{word1}\" in the graph!"

        if word2:
            word2 = word2.lower()
            if word2 not in self.words:
                return f"No \"{word2}\" in the graph!"

            # 使用Dijkstra算法计算最短路径
            distances = {word: float('inf') for word in self.words}
            distances[word1] = 0
            previous = {word: None for word in self.words}
            visited = set()

            heap = [(0, word1)]

            while heap:
                current_dist, current_word = heapq.heappop(heap)

                if current_word in visited:
                    continue

                visited.add(current_word)

                if current_word == word2:
                    break

                for neighbor, weight in self.graph[current_word].items():
                    distance = current_dist + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_word
                        heapq.heappush(heap, (distance, neighbor))

            if distances[word2] == float('inf'):
                return f"No path from \"{word1}\" to \"{word2}\"!"

            # 重建路径
            path = []
            current = word2
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()

            # 可视化路径
            self._visualize_path(path)

            path_str = " -> ".join(path)
            return f"最短路径: {path_str}\n路径长度: {distances[word2]}"
        else:
            # 计算到所有其他节点的最短路径
            results = []
            for target in sorted(self.words):
                if target != word1:
                    result = self.calc_shortest_path(word1, target)
                    results.append(result)
            return "\n".join(results)

    def _visualize_path(self, path):
        if len(path) < 2:
            return

        G = nx.DiGraph()

        # 添加所有节点和边
        for from_word in self.graph:
            for to_word, weight in self.graph[from_word].items():
                G.add_edge(from_word, to_word, weight=weight)

        # 创建路径边列表
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)

        # 绘制所有节点和边
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=True)

        # 高亮显示路径上的节点和边
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=1200, node_color='lightgreen')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red', arrows=True)

        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title(f"Shortest Path: {' -> '.join(path)}")
        plt.show()

    def calc_page_rank(self, word=None, top_n=10):
        if not self.page_rank:
            self._init_page_rank()

        if word:
            word = word.lower()
            if word not in self.page_rank:
                return f"No \"{word}\" in the graph!"
            return f"PageRank of \"{word}\": {self.page_rank[word]:.6f}"
        else:
            # 返回PR值最高的top_n个单词
            sorted_pr = sorted(self.page_rank.items(), key=lambda x: x[1], reverse=True)
            result = "PageRank Top {}:\n".format(top_n)
            for i, (w, pr) in enumerate(sorted_pr[:top_n], 1):
                result += f"{i}. {w}: {pr:.6f}\n"
            return result

    def random_walk(self, output_file=None):
        if not self.words:
            return "Graph is empty!"

        # 随机选择起始节点
        current_word = random.choice(list(self.words))
        visited_edges = set()
        walk_path = [current_word]

        while True:
            # 检查是否有出边
            if current_word not in self.graph or not self.graph[current_word]:
                break

            # 随机选择一个出边
            neighbors = list(self.graph[current_word].keys())
            next_word = random.choice(neighbors)
            edge = (current_word, next_word)

            # 检查是否重复边
            if edge in visited_edges:
                break

            visited_edges.add(edge)
            walk_path.append(next_word)
            current_word = next_word

        walk_text = ' '.join(walk_path)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(walk_text)
            return f"Random walk saved to {output_file}\nWalk path: {walk_text}"
        else:
            return f"Random walk path: {walk_text}"


def main():
    print("=== 文本处理与有向图分析程序 ===")

    # 获取文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("请输入文本文件路径: ")

    # 创建图并处理文本
    graph = TextGraph()
    try:
        graph.build_graph(file_path)
        print("\n文本处理完成，有向图已构建。")
    except Exception as e:
        print(f"错误: {e}")
        return

    while True:

        print("\n===== 功能菜单1 =====")

        print("1. 展示有向图")
        print("2. 查询桥接词")
        print("3. 根据桥接词生成新文本")
        print("4. 计算最短路径")
        print("5. 计算PageRank")
        print("6. 随机游走")
        print("0. 退出")

        choice = input("请选择功能 (0-6): ")

        if choice == '0':
            break
        elif choice == '1':
            save_option = input("是否保存图形到文件? (y/n): ").lower()
            if save_option == 'y':
                save_path = input("请输入保存路径 (如 graph.png): ")
                graph.show_directed_graph(save_path)
            else:
                graph.show_directed_graph()
        elif choice == '2':
            word1 = input("请输入第一个单词: ")
            word2 = input("请输入第二个单词: ")
            print(graph.query_bridge_words(word1, word2))
        elif choice == '3':
            text = input("请输入新文本: ")
            new_text = graph.generate_new_text(text)
            print("\n生成的新文本:")
            print(new_text)
        elif choice == '4':
            word1 = input("请输入起始单词: ")
            word2 = input("请输入目标单词 (留空则计算到所有节点的最短路径): ")
            if word2.strip():
                print(graph.calc_shortest_path(word1, word2))
            else:
                print(graph.calc_shortest_path(word1))
        elif choice == '5':
            word = input("请输入要查询的单词 (留空则显示PR值最高的单词): ")
            if word.strip():
                print(graph.calc_page_rank(word))
            else:
                print(graph.calc_page_rank())
        elif choice == '6':
            output_file = input("请输入输出文件路径 (留空则不保存): ")
            if output_file.strip():
                print(graph.random_walk(output_file))
            else:
                print(graph.random_walk())
        else:
            print("无效选择，请重新输入。")


if __name__ == "__main__":
    main()