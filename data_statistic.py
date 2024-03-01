import os
from concurrent.futures import ProcessPoolExecutor, as_completed


setting_file_path = "E:/jupyterNotes/cg_data/system.in.settings"
input_path = "E:/jupyterNotes/cg_data/dump_info"
group_data_output_path = "E:/jupyterNotes/cg_data/group_data/"

group_dict = {}


group_type_dict = {
    ("bpada1A1", "pa1", "bpada1A5"): "1",
    ("bpada1A2", "bpada1A4", "tape1A2"): "2",
    ("bpada1A3"): "3",
    ("mpd1", "tape1A1"): "4",
    ("tape1A3"): "5",
}

atom_num = 0

# 读取原子数
with open(input_path, "r") as f:
    for line in f:
        # 读取原子数
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            atom_num = int(next(f))
            break

group_id = 0
# 打开system.in.settings
with open(setting_file_path) as f:
    for line in f:
        if line.startswith("    group"):
            lst = line.split()
            for key, value in group_type_dict.items():
                if lst[1] in key:
                    group_type = value
            group_atoms = list(lst[3::])

            group_dict[(group_id, group_type)] = group_atoms
            group_id += 1


def removeFile(path):
    if os.path.exists(path):
        os.remove(path)


def atoms_dist(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


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
        d = (dx**2 + dy**2 + dz**2)**0.5
        
        if d > cutoff or d == 0:  # 跳过距离大于cutoff的原子和自身
            continue
        
        # 计算每个方向上的库仑力分量
        force_magnitude = 332.0636 * q * atoms_q / d**2  # 单位kcal/mol/ai
        for direction in range(3):
            # 计算方向分量
            direction_vector = [dx / d, dy / d, dz / d]
            cf_sum[direction] += force_magnitude * direction_vector[direction]

    return cf_sum

# def get_Coulomb_force_sum(atoms_dict, atom_id, cutoff):
#     cf_sum = [0, 0, 0]
#     x = atoms_dict[atom_id][2]
#     y = atoms_dict[atom_id][3]
#     z = atoms_dict[atom_id][4]
#     q = atoms_dict[atom_id][1]
#     for id, val in atoms_dict.items():
#         a_x = val[2]
#         a_y = val[3]
#         a_z = val[4]
#         d = atoms_dist(x, y, z, a_x, a_y, a_z)
#         if id == atom_id or d > cutoff:
#             continue
#         atoms_q = val[1]
#         for direction in range(3):
#             # 求两个原子间的库伦力在direction方向的分力,direction = 1,2,3分别代表x,y,z方向
#             f_direction = atoms_dict[atom_id][2 + direction] - val[2 + direction]
#             cf_sum[direction] += (332.0636 * q * atoms_q / d**2) * (f_direction / d)
#     return cf_sum


# 生成指定timestep的group_data
def gen_group_data(timeStep):
    output_file_name = "group_data_" + str(timeStep) + ".txt"
    output_file_path = os.path.join(group_data_output_path, output_file_name)

    with open(output_file_path, "w") as g:
        g.write("NUMBER OF GROUPS: \n" + str(len(group_dict)) + "\n")

        with open(input_path, "r") as f:
            for line in f:
                # 读取TIMESTEP
                if line.startswith("ITEM: TIMESTEP"):
                    TIMESTEP = next(f)
                    if int(TIMESTEP) == timeStep:
                        print(TIMESTEP)
                        g.write("TIMESTEP: \n" + TIMESTEP)
                        g.write("id type x y z fx fy fz\n")
                        # 跳到原子数据开始的地方
                        while True:
                            line = next(f)
                            if line.startswith("ITEM: ATOMS"):
                                break
                        # 开始处理当前时间步的原子数据
                        atoms_dict = {}
                        for i in range(atom_num):
                            line = next(f)
                            lst = line.split()
                            atoms_id = lst[0]
                            # mass q x y z fx fy fz
                            atoms_data = list(map(float, lst[-8::]))
                            atoms_dict[atoms_id] = atoms_data

                        # 开始统计每个group数据,并写入文件
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
                                group_x = sum_x / total_mass
                                group_y = sum_y / total_mass
                                group_z = sum_z / total_mass
                            else:
                                print("error: total_mass <= 0")
                                group_x = group_y = group_z = 0  # 避免除以零错误
                            g.write(
                                str(key[0])
                                + " "
                                + str(key[1])
                                + " "
                                + str(group_x)
                                + " "
                                + str(group_y)
                                + " "
                                + str(group_z)
                                + " "
                                + str(group_fx)
                                + " "
                                + str(group_fy)
                                + " "
                                + str(group_fz)
                                + "\n"
                            )


def main():
    minTimeStep = 200000
    maxTimeStep = 201000
    num_processes = 1  # 根据您的机器情况调整进程数

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
        gen_group_data(tsp)


if __name__ == "__main__":
    main()
