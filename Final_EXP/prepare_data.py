import random
import math
import clean_data, group_data, number_to_words

def prepare_data(file_path1, file_path2, group_option=0, time_group=5, num2word_option=0):
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

    def apply_group_data(dataset):
            # Check if the dataset is empty
        if len(dataset) == 0:
           return [], []
        else:
            unseen, seen = random_split(dataset, math.floor(0.3 * len(dataset)))
            return group_data.group_data(unseen, time_group), group_data.group_data(seen, time_group)

    if (group_option == 1):
        datasets = [dev1, dev2]

        del dev1
        del dev2
        dev1_unseen, dev1_seen = apply_group_data(datasets[0])
        dev2_unseen, dev2_seen = apply_group_data(datasets[1])
    else:
        dev1_unseen, dev1_seen = random_split(dev1, math.floor(0.3 * num_elements[0]))
        dev2_unseen, dev2_seen = random_split(dev2, math.floor(0.3 * num_elements[1]))
        
        del dev1
        del dev2

    if num2word_option == 1:
        new_data = [dev1_seen, dev1_unseen, dev2_seen, dev2_unseen]
        for i in range(len(new_data)):
            new_data[i] = number_to_words.convert_numericals_to_words(new_data[i])
        
        # Unpacking the modified lists back to their original variables
        dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = new_data
        del new_data

    print('\033[92mData prepared successfully ✔\033[0m')
    return dev1_seen, dev1_unseen, dev2_seen, dev2_unseen