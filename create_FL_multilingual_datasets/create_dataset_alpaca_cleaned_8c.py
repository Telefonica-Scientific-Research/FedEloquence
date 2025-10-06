import json

# This code creates a jsonl file with data from all languages and the correct distribution for the server and independent clients
# It does so from jsonl files of each language

# List of language labels to identify each client
langs = [
    "ca",  # Catalan
    "da",  # Danish
    "de",  # German
    "en",  # English
    "es",  # Spanish
    "eu",  # Basque
    "hr",  # Croatian
    "sr",  # Serbian
]

# Define the path to the directory containing the JSONL files
jsonl_dir = "/create_FL_multilingual_datasets/alpaca_cleaned/jsonls"

n_clients = len(langs)  # Number of independent clients, one for each language

length_json = 52002 # Total number of sentences per language

print("Number of clients: ", n_clients)

total_length = length_json * n_clients

# Total size of validation and test sets in the server
length_val_server = n_clients * 501 # Length of validation set in the server, counting all languages
length_test_server = n_clients * 501 # Length of validation set in the server, counting all languages

# These values should be the same as the ones in the config file
# They refer to the proportions of the training, validation and test sets of each client
split_train = 0.96 
split_test = 0.02
split_val= 0.02
 
# Total size of the training, validation and test sets of all clients together
length_clients = total_length - length_val_server - length_test_server
print("Size clients: ", length_clients)

n_samp_x_lang_in_val_server = length_val_server / n_clients
print("Samples per language in valid of server: ", n_samp_x_lang_in_val_server)
n_samp_x_lang_in_test_server = length_test_server / n_clients
print("Samples per language in test of server: ", n_samp_x_lang_in_test_server)

n_samp_x_lang_to_train = (split_train * length_clients)/n_clients
print("Samples per language to train one client", n_samp_x_lang_to_train)

n_samp_x_lang_to_val_training = (split_val * length_clients)/n_clients
print("Samples per language for valid one client", n_samp_x_lang_to_val_training)

n_samp_x_lang_to_test_training = (split_test * length_clients)/n_clients
print("Samples per language for test one client", n_samp_x_lang_to_val_training)

def extract_and_concatenate_jsonl(files, sections):
    concatenated_data = []
    
    # Read all lines from each file and store in separate variables
    file_lines = []
    for file in files:
        with open(f"./create_FL_multilingual_datasets/alpaca_cleaned/jsonls/{file}", 'r') as f:
            file_lines.append(f.readlines())
    
    # Extract and concatenate lines for each section
    for section in sections:
        section_part = []
        for i in range(len(file_lines)):
            section_part.extend(file_lines[i][:section])
            file_lines[i] = file_lines[i][section:]  # Update the lines to remove the extracted part
        concatenated_data.extend(section_part)
    
    # Create new JSONL data with 'instruction', 'input', 'output' and 'domain' fields (in domain we store the language label)
    new_data = []
    for line in concatenated_data:
        data_dict = json.loads(line)
        new_data_dict = {
            "instruction": data_dict["instruction"],
            "input": data_dict["input"] if "input" in data_dict else "",  # Handle cases where 'input' might not exist
            "output": data_dict["output"],
            "domain": data_dict["domain"]
        }
        new_data.append(new_data_dict)
    
    return new_data

files = ["ca.jsonl", "da.jsonl", "de.jsonl", "en.jsonl", "es.jsonl", "eu.jsonl", "hr.jsonl", "sr.jsonl"]

# Number of samples per language in each partition of the dataset
sections = [int(n_samp_x_lang_in_val_server), int(n_samp_x_lang_in_test_server), int(n_samp_x_lang_to_train), int(n_samp_x_lang_to_val_training), int(n_samp_x_lang_to_test_training)]  # Define the number of lines for each section
new_data = extract_and_concatenate_jsonl(files, sections)
print("Number of samples per language in each partition of the dataset: ", sections)

# Save the new JSONL data to a new file
with open('./data/alpaca_cleaned_8c.jsonl', 'w') as output_file:
    for entry in new_data:
        output_file.write(json.dumps(entry) + '\n')