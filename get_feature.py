import os
from concurrent.futures import ProcessPoolExecutor, as_completed

CUTOFF = 20
output_path = "cg_data/fi_i_test/"


# 求两个group的距离，单位 埃
def dist(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


# 得到xyz方向上的fi_i, type是group1的type, group1是要计算fi_i的group，direction是xyz方向，
def get_fi_i(type, group_dict, group1, direction):
    # 创建字典，按照不同的type-type，存储195个fi_i
    fi_dict = {}
    for i in range(1, 6):
        for j in range(i, 6):
            key = f"{i}-{j}"
            fi_dict[key] = [0] * 13
    # target:fx/fy/fz
    fi_dict["target"] = group1[direction + 3]

    # L 1~5
    for L in range(1, 6):
        # k -12~0
        for k in range(-12, 1):
            sum_fi = 0
            for group2 in group_dict[L]:
                d = dist(
                    group1[1], group1[2], group1[3], group2[1], group2[2], group2[3]
                )
                # 如果两个group是同一个group或者距离大于cutoff，跳过
                if group1[0] == group2[0] or d > CUTOFF:
                    continue
                else:
                    # 统计feature，xij * Rij^(k-1)
                    print(pow(d, k-1))
                    sum_fi += (group1[direction] - group2[direction]) * pow(d, k-1)
            if L >= type:
                fi_dict[f"{type}-{L}"][12 + k] = sum_fi
            else:
                fi_dict[f"{L}-{type}"][12 + k] = sum_fi
    return fi_dict


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def save_dict_to_file(fi_dict, file_path):
    with open(file_path, "a") as file:
        for key in fi_dict:
            if key == "target":
                file.write(str(fi_dict[key]) + ";")
            else:
                file.write(str(fi_dict[key])[1:-1] + "; ")
        file.write("\n")


# 得到指定TIMESTEP的所有group的fi_i
def get_all_fi(group_dict, tsp):
    file_path = output_path + str(tsp) + ".txt"
    remove_file(file_path)
    for type in range(1, 6):
        for group1 in group_dict[type]:
            # 分别计算每个group三个方向上的fi_i
            dic_x = get_fi_i(type, group_dict, group1, 1)
            save_dict_to_file(dic_x, file_path)
            dic_y = get_fi_i(type, group_dict, group1, 2)
            save_dict_to_file(dic_y, file_path)
            dic_z = get_fi_i(type, group_dict, group1, 3)
            save_dict_to_file(dic_z, file_path)


def process(tsp):
    # 按照type对group分组
    input_groupData_path = ("cg_data/group_data/group_data_" + str(tsp) + ".txt")
    with open(input_groupData_path, "r") as file:
        for line in file:
            if line.startswith("NUMBER OF GROUPS:"):
                num_groups = int(next(file))
            if line.startswith("TIMESTEP:"):
                TIMESTEP = int(next(file))
                if TIMESTEP == tsp:
                    # 按照类型存储不同的group
                    group_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
                    print("TIMESTEP: " + str(TIMESTEP))
                    next(file)
                    for i in range(num_groups):
                        line = next(file)
                        lst = line.split()
                        # 把每个group按照id x y z fx fy fz的格式包装成一个tuple，加到对应type的值list里， x单位埃，fx单位kcal/mol/ai
                        group_dict[int(lst[1])].append((int(lst[0]), float(lst[2]), float(lst[3]), float(lst[4]),
                                                        float(lst[5]), float(lst[6]), float(lst[7]),))
                    get_all_fi(group_dict, TIMESTEP)


def main():
    minTimeStep = 0
    maxTimeStep = 20000
    num_processes = 1  # 根据您的机器情况调整进程数, 要确保(maxTimeStep - minTimeStep)能够整除num_processes

    # 计算每个进程要处理的时间步数量
    steps_per_process = (maxTimeStep - minTimeStep) // num_processes

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            start = minTimeStep + i * steps_per_process
            end = start + steps_per_process if i < num_processes - 1 else maxTimeStep
            # 每个进程处理一个时间步区间
            futures.append(executor.submit(process_range, start, end))


def process_range(start, end):
    for tsp in range(start, end, 1000):
        process(tsp)


if __name__ == "__main__":
    main()
