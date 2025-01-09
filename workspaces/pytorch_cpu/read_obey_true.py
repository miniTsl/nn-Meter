import json

file_path = "fusion_rule_test/results/detected_fusion_rule.json"
with open(file_path, 'r') as f:
    profiled_results = json.load(f)
    obey_tests = []
    disobey_tests = []
    for key, value in profiled_results.items():
        if value["obey"] == True:
            obey_tests.append(key[3:])
        else:
            disobey_tests.append(key[3:])
    # in rising order
    obey_tests.sort()
    disobey_tests.sort()
    # print one by one, each in a new line
    print("obey tests:")
    for test in obey_tests:
        print(test)
    print()
    print("disobey tests:")
    for test in disobey_tests:
        print(test)