import pandas as pd
import ipaddress
import os
def clean_data(target_file):
    print("Cleaning data... for " + os.path.split(target_file)[1])

    # Step 1: Read Data in Chunks
    chunk_size = 10000  # Adjust the chunk size based on your system's memory
    filtered_data_chunks = pd.read_json(target_file, lines=True, chunksize=chunk_size)

    # Step 2: Process Each Chunk
    output = []

    for chunk in filtered_data_chunks:
        # Handle cases where 'flows' might contain a float instead of a dictionary
        chunk['destinationIPv4Address'] = chunk['flows'].apply(lambda x: x.get('destinationIPv4Address') if isinstance(x, dict) else None)
        
        # Filter out unwanted IP addresses in one step
        chunk = chunk[~chunk['destinationIPv4Address'].apply(lambda x: isinstance(x, str) and ':' not in x)]
        chunk = chunk[~chunk['destinationIPv4Address'].apply(lambda x: isinstance(ipaddress.IPv4Address(x), (ipaddress.IPv4Multicast, ipaddress.IPv4Private)))]

        # Concatenate the 'flows' lists
        output.extend(chunk['flows'].tolist())

    # Flatten and clean the output
    output = [str(item).replace(',', ' ').replace('}', '').replace('{', '').replace("]", '').replace("[", '').replace("'", '').strip('[]') for item in output]
    output = [[s] for s in output if s]
    
    num_elements = len(output)
    print(f"{num_elements} flows!")

    return output, num_elements
