import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from pandas import DataFrame

# setting_file_path = "cg_data/system.in.settings"
# origin_data_input_path = "cg_data/dump_info"

setting_file_path = "cg_data/test_setting"
origin_data_input_path = "cg_data/test_in"

features_data_output_path = "cg_data/features/"

group_dict = {}  # 键:(id，type), 值:[这个group所有的原子id]

group_type_dict = {  # group名称与group类型对应关系
    ("bpada1A1", "pa1", "bpada1A5"): "1",
    ("bpada1A2", "bpada1A4", "tape1A2"): "2",
    ("bpada1A3"): "3",
    ("mpd1", "tape1A1"): "4",
    ("tape1A3"): "5",
}

atom_num = 0

# 读取原子数
with open(origin_data_input_path, "r") as f:
    for line in f:
        # 读取原子数
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            atom_num = int(next(f))
            break

group_id = 0
# 打开system.in.settings, 读取所有group信息，每个group生成一个id，存到group_dict里
with open(setting_file_path) as f:
    for line in f:
        if line.startswith("    group"):
            lst = line.split()
            for key, value in group_type_dict.items():
                if lst[1] in key:
                    group_type = int(value)
            group_atoms = map(int, list(lst[3::]))

            group_dict[(group_id, group_type)] = group_atoms
            group_id += 1


# 删除指定文件名文件
def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


# 返回两个坐标的距离，单位：埃
def dist(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


# ----------------------------------------------------------------------------------------------------------------------------

CUTOFF = 20


# 统计一个group在xyz其中一个方向上的feature, type是group1的type, group1是要计算feature的group，direction是选择xyz方向，
def get_one_direction_features_of_one_group(type_of_group1, dict_of_groups, group1, direction):
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
            for group2 in dict_of_groups[L]:
                d = dist(
                    group1[1], group1[2], group1[3], group2[1], group2[2], group2[3]
                )
                # 如果两个group是同一个group或者距离大于cutoff，跳过
                if group1[0] == group2[0] or d > CUTOFF:
                    continue
                else:
                    # 统计feature，xij * Rij^(k-1)
                    sum_fi += (group1[direction] - group2[direction]) * pow(d, k - 1)
            if L >= type_of_group1:
                fi_dict[f"{type_of_group1}-{L}"][12 + k] = sum_fi
            else:
                fi_dict[f"{L}-{type_of_group1}"][12 + k] = sum_fi
    data = []
    for k,v in fi_dict.items():
        if k != "target":
            for feature in v:
                data.append(feature)
        else:
            data.append(v)
    return data


# 得到指定时间步的features
def get_features_of_this_timestep(dict_of_groups, tsp):
    file_path = features_data_output_path + str(tsp)+".csv"
    remove_file(file_path)

    all_features = []
    for Type in range(1, 6):
        for group1 in dict_of_groups[Type]:
            # 分别计算每个group三个方向上的feature
            feature_x  = get_one_direction_features_of_one_group(Type, dict_of_groups, group1, 1)
            feature_y = get_one_direction_features_of_one_group(Type, dict_of_groups, group1, 2)
            feature_z = get_one_direction_features_of_one_group(Type, dict_of_groups, group1, 3)
            all_features.append(feature_x)
            all_features.append(feature_y)
            all_features.append(feature_z)

    df = pd.DataFrame(all_features)
    column_titles = [str(i) for i in range(len(df.columns) - 1)] + ['target']
    df.columns = column_titles

    df.to_csv(file_path, index=False)


# ----------------------------------------------------------------------------------------------------------------------------


# 求一个原子三个方向上所受的库伦力分量之和
def get_Coulomb_force_sum(atoms_dict, atom_id, cutoff):
    cf_sum = [0.0, 0.0, 0.0]  # 用于存储x,y,z方向上的力
    x, y, z, q = atoms_dict[atom_id][2], atoms_dict[atom_id][3], atoms_dict[atom_id][4], atoms_dict[atom_id][1]

    for id, val in atoms_dict.items():
        if id == atom_id:
            continue  # 跳过自身
        a_x, a_y, a_z, atoms_q = val[2], val[3], val[4], val[1]

        # 计算距离
        dx, dy, dz = x - a_x, y - a_y, z - a_z
        d = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        if d > cutoff or d == 0:  # 跳过距离大于cutoff的原子和自身
            continue

        # 计算每个方向上的库仑力分量
        force_magnitude = 332.0636 * q * atoms_q / d ** 2  # 单位kcal/mol/ai
        for direction in range(3):
            # 计算方向分量
            direction_vector = [dx / d, dy / d, dz / d]
            cf_sum[direction] += force_magnitude * direction_vector[direction]

    return cf_sum


# 统计指定时间步的group的数据，生成features后保存
def gen_features_and_save(timeStep):
    with open(origin_data_input_path, "r") as reader:
        for line in reader:
            # 读取TIMESTEP
            if line.startswith("ITEM: TIMESTEP"):
                timestep_at_now = int(next(reader))
                if timestep_at_now == timeStep:
                    print(timestep_at_now)

                    # 跳到原子数据开始的地方
                    while True:
                        line = next(reader)
                        if line.startswith("ITEM: ATOMS"):
                            break

                    # 开始读取当前时间步的所有原子数据
                    atoms_dict = {}  # 键:id, 值:mass q x y z fx fy fz
                    for i in range(atom_num):
                        line = next(reader)
                        lst = line.split()
                        atoms_id = int(lst[0])
                        # mass q x y z fx fy fz ，x单位埃， fx单位kcal/mol/ai
                        atoms_data = list(map(float, lst[-8::]))
                        atoms_dict[atoms_id] = atoms_data

                    # 按照类型存储不同的group
                    dict_of_5_types_group = {1: [], 2: [], 3: [], 4: [], 5: []}
                    
                    # 开始统计每个group的数据,统计完成后存放进对应type的list里
                    for key, value in group_dict.items():
                        group_fx = 0  # 该group内所有原子所受力的合力
                        group_fy = 0
                        group_fz = 0

                        total_mass = 0  # 累加该组内所有原子的总质量
                        sum_x = 0  # 累加质量乘以x坐标
                        sum_y = 0  # 累加质量乘以y坐标
                        sum_z = 0  # 累加质量乘以z坐标

                        for atoms_id in value:
                            atoms_mass = atoms_dict[atoms_id][0]
                            atoms_x = atoms_dict[atoms_id][2]
                            atoms_y = atoms_dict[atoms_id][3]
                            atoms_z = atoms_dict[atoms_id][4]

                            sums = get_Coulomb_force_sum(
                                atoms_dict, atoms_id, 12
                            )


                            # group_fx单位 kcal/mol/ai
                            group_fx += atoms_dict[atoms_id][5]
                            group_fx -= sums[0]
                            group_fy += atoms_dict[atoms_id][6]
                            group_fy -= sums[1]
                            group_fz += atoms_dict[atoms_id][7]
                            group_fz -= sums[2]

                            # 累加计算质心所需的值
                            total_mass += atoms_mass
                            sum_x += atoms_x * atoms_mass
                            sum_y += atoms_y * atoms_mass
                            sum_z += atoms_z * atoms_mass

                        # 确保计算质心前总质量大于0
                        if total_mass > 0:
                            # group_x单位埃
                            group_x = sum_x / total_mass
                            group_y = sum_y / total_mass
                            group_z = sum_z / total_mass
                        else:
                            print("error: total_mass <= 0")
                            group_x = group_y = group_z = 0  # 避免除以零错误

                        dict_of_5_types_group[key[1]].append(  # id x y z fx fy fz
                            (key[0], group_x, group_y, group_z, group_fx, group_fy, group_fz))

                    # 生成当前时间步的features
                    get_features_of_this_timestep(dict_of_5_types_group, timestep_at_now)


# def main():
#     minTimeStep = 0
#     maxTimeStep = 1000
#     num_processes = 1  # 根据您的机器情况调整进程数, 要确保(maxTimeStep - minTimeStep)能够整除num_processes
#
#     # 计算每个进程要处理的时间步数量
#     steps_per_process = (maxTimeStep - minTimeStep) // num_processes
#
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = []
#         for i in range(num_processes):
#             start = minTimeStep + i * steps_per_process
#             end = start + steps_per_process if i < num_processes - 1 else maxTimeStep
#             # 每个进程处理一个时间步区间
#             futures.append(executor.submit(process_range, start, end))
#
#
# def process_range(start, end):
#     for tsp in range(start, end, 1000):
#         gen_features_and_save(tsp)
#
#
# if __name__ == "__main__":
#     main()

gen_features_and_save(0)