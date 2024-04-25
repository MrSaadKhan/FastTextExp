import random
import math
import clean_data

def prepare_data(file_path1, file_path2):
    file_path = [file_path1, file_path2]
    num_elements = []
    data = []

    for file in file_path:
        temp, temp1 = clean_data.clean_data(file)
        data.append(temp)  # Append each piece of data to the list
        num_elements.append(temp1)

    def split_list(lst, split_index):
        return lst[:split_index], lst[split_index:]
    
    flattened_data = []
    # for sublist in data:
    #     for subsublist in sublist:
    #         flattened_data.extend(subsublist)

    

    def flatten(data):
        for sublist in data:
            flattened_data.extend(sublist)
        return flattened_data

    flattened_data = flatten(data)

    dev1, dev2 = split_list(flattened_data, num_elements[0])
    del data, flattened_data

    # def random_split(lst, n):
    #     selected_items = random.sample(lst, n)
    #     remaining_items = [item for item in lst if item not in selected_items]
    #     return selected_items, remaining_items
    
    # def random_split(lst, n):
    #     selected_indices = random.sample(range(len(lst)), n)
    #     selected_items = [lst[i] for i in selected_indices]
    #     remaining_items = [item for idx, item in enumerate(lst) if idx not in selected_indices]
    #     return selected_items, remaining_items

    def random_split(lst, n):
        selected_indices = set(random.sample(range(len(lst)), n))
        selected_items = [lst[i] for i in selected_indices]
        remaining_items = [lst[i] for i in range(len(lst)) if i not in selected_indices]
        return selected_items, remaining_items


    dev1_unseen, dev1_seen = random_split(dev1, math.floor(0.3 * num_elements[0]))
    del dev1

    dev2_unseen, dev2_seen = random_split(dev2, math.floor(0.3 * num_elements[1]))
    del dev2

    print('\033[92mData prepared successfully ✔\033[0m')
    return dev1_seen, dev1_unseen, dev2_seen, dev2_unseen