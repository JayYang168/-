'''
@Description: 
@Author: yy
@Date: 2024-09-01 10:38:49
@LastEditTime: 2024-09-01 10:52:08
@LastEditors: yy
'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['simHei']  # 中文显示

class Node:
    def __init__(self,code:int,x:float,y:float) -> None:
        self.code = code
        self.x = x
        self.y = y
        
    def __repr__(self) -> str:
        return f'({self.code},{self.x},{self.y})'
    

class SimulatedAnnealing:
    # 模拟退火算法
    def __init__(self,initial_temp:float,cooling_rate:float,num_iterations:int) -> None:
        self.initial_temp = initial_temp         # 初始温度
        self.cooling_rate = cooling_rate         # 降温系数
        self.num_iterations = num_iterations     # 迭代次数
    
    # 定义邻域生成函数
    def two_opt(self,x):
        x_new = x.copy()
        i, j = np.random.choice(len(x), size=2, replace=False)
        if i > j:
            i,j = j,i
        x_new[i:j] = x_new[i:j][::-1]
        return x_new

    def search(self,nodes,objfunc):
        current_solution = np.arange(len(nodes))
        best_solution = current_solution.copy()
        current_temp = self.initial_temp
        best_distance = objfunc(current_solution)
        
        distances = [best_distance]

        for _ in range(self.num_iterations):
            new_solution = self.two_opt(current_solution)
            new_distance = objfunc(new_solution)
            
            # 判断是否接受新的解
            if new_distance < best_distance:
                best_solution = new_solution.copy()
                best_distance = new_distance
            elif np.random.rand() < np.exp((objfunc(current_solution) - new_distance) / current_temp):
                current_solution = new_solution.copy()
            
            current_temp *= self.cooling_rate # 降温
            distances.append(best_distance)

        return best_solution, best_distance, distances

def objfunc(x:np.ndarray):
    s = 0.0
    for i,j in zip(x[:-1],x[1:]):
        s += dis[i,j]
    s += dis[x[-1],x[0]]
    return s
### 算例 ###
file = r'berlin52.txt'
nodes = {}
with open(file,'r') as f:
    contents = f.readlines()[6:-1]
    for i,content in enumerate(contents):
        info = content.strip().split(' ')
        node = Node(int(info[0]),float(info[1]),float(info[2]))
        nodes[i] = node
nodes_num = len(nodes)
dis = np.zeros((nodes_num,nodes_num))
for i in range(nodes_num):
    for j in range(nodes_num):
        dis[i,j] = np.sqrt((nodes[i].x-nodes[j].x)**2+(nodes[i].y-nodes[j].y)**2)

### 算法 ###
initial_temp = 1000
cooling_rate = 0.9988
num_iterations = 10000
np.random.seed(42)
alg = SimulatedAnnealing(initial_temp,cooling_rate,num_iterations)
# 运行模拟退火算法
best_solution, best_distance, distances = alg.search(nodes,objfunc)

# 打印结果
print("最优路径:", '->'.join([str(i) for i in best_solution] + [str(best_solution[0])]))
print("最短路径长度:", best_distance)



xs = [nodes[i].x for i in best_solution]
ys = [nodes[i].y for i in best_solution]
xs.append(xs[0])
ys.append(ys[0])
# 画出路径
plt.figure(figsize=(10, 5))

# # 最优路径图
plt.subplot(1, 2, 1)
plt.plot(xs,ys,
         marker='o')
plt.title('搜索路径结果')
plt.xlabel('X')
plt.ylabel('Y')
# 距离收敛图
plt.subplot(1, 2, 2)
plt.plot(distances)
plt.title('距离收敛曲线')
plt.xlabel('迭代次数')
plt.ylabel('距离')

plt.tight_layout()
plt.savefig('SA结果.png',dpi=600,bbox_inches='tight')
plt.show()

