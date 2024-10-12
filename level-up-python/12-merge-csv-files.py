import csv

parent_path='level-up-python-3210418-main/src/12 Merge CSV Files'

def merge_csv(list_of_files, output_file):
    field_set=list()
    for file in list_of_files:
        with open(f"{parent_path}/{file}", "r") as infile:
            reader=csv.DictReader(infile)
            field_set.extend(f for f in reader.fieldnames if f not in field_set)

    with open(output_file, "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=field_set)
        writer.writeheader()

        for file in list_of_files:
            with open(f"{parent_path}/{file}", "r") as infile:
                reader=csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)

# commands used in solution video for reference
if __name__ == '__main__':
    merge_csv(['class1.csv', 'class2.csv'], 'all_students.csv')
