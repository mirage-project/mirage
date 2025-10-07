import sys

def deduplicate_log(log_file: str):
    output_file = log_file + "_deduplicated.txt"
    lines_set = set()
    with open(log_file, "r") as input_f:
        with open(output_file, "w") as output_f:
            lines = input_f.readlines()
            for line in lines:
                if line not in lines_set:
                    lines_set.add(line)
                    print(line)
                    output_f.write(line)
            

if __name__ == "__main__":
    deduplicate_log(sys.argv[1])