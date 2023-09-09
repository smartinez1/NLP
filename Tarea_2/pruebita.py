import os
import xml.etree.ElementTree as ET
import re

# Define the path to the BAC dataset folder
path_BAC = "Datasets\\BAC"

# Define the path to the consolidated text file
consolidated_BAC_file = "consolidated_BAC.txt"

# Function to consolidate text from XML documents in a folder
def consolidate_text_from_xml(folder_path, output_file):
    with open(output_file, "w", encoding="utf-8") as output:
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # Check if the file is within the "blogs" folder and has a ".xml" extension
                if "blogs" in root and file_name.endswith(".xml"):
                    try:
                        tree = ET.parse(file_path)
                        root_element = tree.getroot()
                        for post_element in root_element.findall(".//post"):
                            content = post_element.text.strip() if post_element.text else ""
                            # Write the extracted content to the output file
                            output.write(content)
                            output.write("\n")  # Add a newline to separate documents
                    except ET.ParseError as e:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_:
                            xml_content = file_.read()

                        # Define a regular expression pattern to match content between <date> and </date> tags
                        pattern = r'<date>.*?</date>'

                        # Use re.sub() to replace matched patterns with an empty string
                        xml_content_filtered = re.sub(pattern, '', xml_content, flags=re.DOTALL)

                        xml_content_filtered = re.sub(r'<.*?>', '', xml_content_filtered)
                        # Remove backslashes before single quotes
                        xml_content_filtered = re.sub(r'(?<=\\)\'', '\'', xml_content_filtered)

                        xml_content_filtered = xml_content_filtered.replace('\n', '')
                        output.write(xml_content_filtered)
                        output.write("\n")  # Add a newline to separate documents
                        print(f"Error parsing {file_path}: {e}, using raw file...")
                        continue

# Path to the "blogs" folder within the BAC dataset
blogs_path_BAC = os.path.join(path_BAC, "blogs")

# Consolidate text content from XML documents within the "blogs" folder of the BAC dataset
consolidate_text_from_xml(blogs_path_BAC, consolidated_BAC_file)
print("Consolidated BAC dataset (text content from XML documents within 'blogs' folder).")