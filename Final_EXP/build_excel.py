import re

def remove_non_numeric_and_m(text):
    # Define a regex pattern to match non-numeric characters and 'm's
    pattern = r'[^0-9.m\n]'
    # Replace all non-numeric characters and 'm's with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def remove_leading_m_from_new_lines(text):
    # Define a regex pattern to match new lines starting with 'm'
    pattern = r'\nm'
    # Replace 'm' at the beginning of new lines with an empty string
    cleaned_text = re.sub(pattern, '\n', text)
    return cleaned_text

def extract_decimal_numbers_between_m(text):
    # Define a regex pattern to match decimal numbers between 'm's
    pattern = r'm([\d.]+)m'
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    # Join the matched numbers into a single string
    result = '\n'.join(matches)
    return result

def remove_zero_and_period_new_lines(text):
    # Define a regex pattern to match new lines containing only '0' or '.'
    pattern = r'(^|\n)[0.\s]*(?=\n|$)'
    # Remove new lines containing only '0' or '.', preserving the number of lines
    cleaned_text = re.sub(pattern, r'\1', text)
    return cleaned_text

# Initialize a list to store line numbers
empty_line_numbers = []

# Open the input file (test.txt) and process each line
with open('test.txt', 'r') as file:
    for line_number, line in enumerate(file, start=1):
        if line.strip() == "":
            # Add the line number to the list
            empty_line_numbers.append(line_number)

# Increment each line number by 1
empty_line_numbers = [num + 1 for num in empty_line_numbers]

# Open the input file
with open('output.txt', 'r') as file:
    # Read the content of the file
    file_content = file.read()

# Remove all non-numeric characters and 'm's from the file content
cleaned_content = remove_non_numeric_and_m(file_content)

# Remove 'm' at the beginning of new lines
cleaned_content = remove_leading_m_from_new_lines(cleaned_content)

# Extract decimal numbers between 'm's from the cleaned content
extracted_numbers = extract_decimal_numbers_between_m(cleaned_content)

# Remove new lines containing only '0' or '.', preserving the number of lines
final_content = remove_zero_and_period_new_lines(extracted_numbers)

# Write the final content to the test.txt file
with open('test.txt', 'w') as output_file:
    output_file.write(final_content)

# Initialize an empty list to store modified lines
modified_lines = []

# Open test.txt to read the lines
with open('test.txt', 'r') as file:
    # Read all the lines
    lines = file.readlines()

# Iterate through the empty line numbers and select the corresponding lines
for line_number in empty_line_numbers:
    # Ensure the line number is within the range of lines
    if 0 < line_number <= len(lines):
        # Remove the last digit from the line
        cleaned_line = lines[line_number - 1].strip()[:-1]
        # Add the cleaned line to the modified lines list
        modified_lines.append(cleaned_line)

# Print the modified lines for debugging
print("Modified lines from test.txt:")
for modified_line in modified_lines:
    print(modified_line)

# Initialize an empty list to store matched lines from output.txt
matched_lines = []

# Open output.txt to read the lines
with open('output.txt', 'r') as file:
    # Read all the lines
    output_lines = file.readlines()

# Iterate through the modified lines from test.txt
for modified_line in modified_lines:
    # Iterate through the lines in output.txt
    for output_line in output_lines:
        # Check if the modified line from test.txt is a substring of the line in output.txt
        if re.search(modified_line, output_line):
            # Add the matched line from output.txt to the list
            matched_lines.append(output_line)
            # Print the matched lines for debugging
            print("Matched line from output.txt:", output_line)
            break  # Exit the inner loop after finding a match

# Write the matched lines to test2.txt
with open('test2.txt', 'w') as file:
    # Write the matched lines to test2.txt
    for line in matched_lines:
        file.write(line)
