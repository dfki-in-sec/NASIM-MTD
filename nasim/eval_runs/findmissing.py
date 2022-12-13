import os


if __name__ == "__main__":
    lr_list = [0.5, 0.1, 0.001]
    batch_size = [32, 64]
    replay_size = [10000, 100000]
    final_epsilon_list = [0.3, 0.05, 0.001]
    exploration_steps_list = [500, 10000, 20000]  # for 200000 train steps
    gamma_list = [0.4, 0.9, 0.999]
    hidden_sizes = [[64, 64], [128, 128]]
    target_update_freq = [100, 1000]
    para_list = []

    for lr in lr_list:
            for batch in batch_size:
                for relay in replay_size:
                    for final_epsilon in final_epsilon_list:
                        for exploration_steps in exploration_steps_list:
                            for gamma in gamma_list:
                                for hidden in hidden_sizes:
                                    for target in target_update_freq:
                                        para_list.append \
                                            (str(lr) + "_" + str( batch) + "_" + str( relay) + "_" + str( final_epsilon) + "_" + str( exploration_steps) + "_" + str( gamma) + "_" + str( hidden) + "_" + str( target))

    directory = "runs"
    # for each agent
    for file in os.listdir(directory):
        list = file.split("_")
        if file[0] == "_":
            para_list.remove(str(list[1]) + "_" + str( list[2]) + "_" + str( list[3]) + "_" + str( list[4]) + "_" + str( list[5]) + "_" + str( list[6]) + "_" + str( list[7]) + "_" + str( list[8]))
    print(para_list)
    print(len(para_list))
    
    
# Output
# ['0.5_64_10000_0.3_10000_0.4_[128, 128]_1000', '0.5_64_10000_0.3_10000_0.999_[128, 128]_1000', '0.5_64_10000_0.3_20000_0.9_[128, 128]_1000', '0.5_64_10000_0.05_500_0.4_[128, 128]_1000', '0.5_64_10000_0.05_500_0.999_[128, 128]_1000', '0.5_64_10000_0.05_10000_0.9_[128, 128]_1000', '0.5_64_10000_0.05_20000_0.4_[128, 128]_1000', '0.5_64_10000_0.05_20000_0.999_[128, 128]_1000', '0.5_64_100000_0.3_500_0.4_[128, 128]_1000', '0.5_64_100000_0.3_500_0.999_[128, 128]_1000', '0.5_64_100000_0.3_10000_0.9_[128, 128]_1000', '0.5_64_100000_0.3_20000_0.4_[128, 128]_1000', '0.5_64_100000_0.3_20000_0.999_[128, 128]_1000', '0.5_64_100000_0.05_500_0.9_[128, 128]_1000', '0.5_64_100000_0.05_10000_0.4_[128, 128]_1000', '0.5_64_100000_0.05_10000_0.999_[128, 128]_1000', '0.5_64_100000_0.05_20000_0.9_[128, 128]_1000', '0.5_64_100000_0.001_500_0.4_[128, 128]_1000', '0.5_64_100000_0.001_500_0.999_[128, 128]_1000', '0.5_64_100000_0.001_10000_0.9_[128, 128]_1000', '0.5_64_100000_0.001_20000_0.4_[128, 128]_1000', '0.5_64_100000_0.001_20000_0.999_[128, 128]_1000']
# 22
